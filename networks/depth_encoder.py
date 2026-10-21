import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import math
import torch.cuda


# =============================================================================
# Neuro-inspired Implicit High-Frequency Feature Extraction Module
# 神经启发式隐式高频特征提取模块
# =============================================================================
# Inspired by visual neuroscience:
# - V1 simple cells: Gabor-like receptive fields for edge/orientation detection
# - Multi-scale processing in visual cortex
# - Frequency-selective modulation in visual pathways
# =============================================================================


class LearnableGaborConv2d(nn.Module):
    """
    可学习 Gabor 卷积层
    
    模拟视觉皮层 V1 区简单细胞的感受野特性。
    通过可学习参数自动调整 Gabor 滤波器的方向、频率等特性。
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=7, 
                 num_orientations=4, num_frequencies=2, stride=1, padding=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_orientations = num_orientations
        self.num_frequencies = num_frequencies
        self.stride = stride
        
        if padding is None:
            padding = kernel_size // 2
        self.padding = padding
        
        self.num_filters = num_orientations * num_frequencies
        
        # 可学习的 Gabor 参数 (用更安全的初始化)
        # theta: 避免 0 和 π 重复方向, 用 [0, π) 均匀分布
        self.theta = nn.Parameter(torch.arange(num_orientations, dtype=torch.float32) * (np.pi / num_orientations))
        self.gamma = nn.Parameter(torch.tensor([0.5] * num_frequencies).clamp(min=0.1, max=0.9))
        self.psi = nn.Parameter(torch.zeros(num_frequencies))
        # sigma/lambda 以 log 空间存储, 初始值落在 clamp 范围内避免卡死
        # exp(0.7)≈2.0 -> sigma; exp(1.4)≈4.0 -> lambda (在 [2,10] 内)
        self.sigma_scale = nn.Parameter(torch.ones(num_frequencies) * 0.7)
        self.lambda_scale = nn.Parameter(torch.ones(num_frequencies) * 1.4)
        
        # 标准卷积层
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)

        # Gabor 投影: 1x1 conv, 移到 __init__ 以保证可学习且稳定
        self.gabor_proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        nn.init.kaiming_normal_(self.gabor_proj.weight, mode='fan_out', nonlinearity='relu')

        # 输出投影 - residual fusion: out = conv_out + proj([conv_out, gabor_out])
        # 最后一层 (BN) 零初始化 weight, 保证初始 proj 输出≈0 -> out≈conv_out (恒等),
        # Gabor 贡献从 0 平滑长起来, 避免随机强注入破坏主干特征
        self.projection = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.01),  # 更保守的 BN
        )
        nn.init.kaiming_normal_(self.projection[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.projection[1].weight)  # 零初始化 -> 初始残差贡献为 0
        nn.init.zeros_(self.projection[1].bias)
        
        # 调制网络 - 初始化为全1来保证不影响输入
        self.modulation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
        )
        nn.init.zeros_(self.modulation[1].weight)  # 零权重 -> 输出1
    
    def _create_gabor_kernel(self, device):
        """创建 Gabor 核"""
        kernels = []
        for f_idx in range(self.num_frequencies):
            sigma = self.sigma_scale[f_idx].exp().clamp(min=0.5, max=5.0)  # 限制范围
            lambda_val = self.lambda_scale[f_idx].exp().clamp(min=2.0, max=10.0)  # 限制范围
            
            for o_idx in range(self.num_orientations):
                theta = self.theta[o_idx]
                
                x = torch.arange(-(self.kernel_size - 1) // 2, 
                               (self.kernel_size - 1) // 2 + 1, 
                               dtype=torch.float32, device=device)
                y = torch.arange(-(self.kernel_size - 1) // 2, 
                               (self.kernel_size - 1) // 2 + 1, 
                               dtype=torch.float32, device=device)
                X, Y = torch.meshgrid(x, y, indexing='ij')
                
                x_theta = X * torch.cos(theta) + Y * torch.sin(theta)
                y_theta = -X * torch.sin(theta) + Y * torch.cos(theta)
                
                gamma = torch.sigmoid(self.gamma[f_idx]).clamp(min=0.3, max=0.8)
                
                gaussian = torch.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2))
                sinusoid = torch.cos(2 * np.pi * x_theta / lambda_val + self.psi[f_idx])
                kernel_real = gaussian * sinusoid
                
                # 去均值: 保证零直流响应, 成为真正的高频/带通滤波器
                kernel_real = kernel_real - kernel_real.mean()
                
                # 归一化核
                kernel_real = kernel_real / (kernel_real.abs().max() + 1e-6)
                
                kernels.append(kernel_real)
        
        return torch.stack(kernels, dim=0)  # [num_filters, H, W]
    
    def forward(self, x):
        device = x.device
        B, C, H, W = x.shape
        
        # 输入清洗 - 就地清洗 NaN/Inf 而非 return x (return x 的通道数/尺寸与正常输出不一致, 会拖垮下游)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 标准卷积处理输入（可能包含下采样）
        conv_out = self.conv(x)  # [B, out_channels, H/stride, W/stride]
        _, _, H_out, W_out = conv_out.shape
        
        # 获取 Gabor 核 - 使用 clamp 限制参数范围
        gabor_kernels = self._create_gabor_kernel(device)  # [num_filters, H, W]
        
        # 对每个滤波器方向分别应用, 再做能量聚合 (energy aggregation)
        # 直接对方向求平均会让相反方向的边界响应相互抵消 -> 相关性≈0
        # 改为对响应平方后跨方向求均值再开方 (L2 能量), 保留方向选择性
        gabor_responses = []
        for f_idx in range(self.num_filters):
            kernel = gabor_kernels[f_idx]  # [H, W]
            # 使用深度卷积模拟 Gabor
            response = F.conv2d(x, kernel.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1),
                               padding=self.padding, groups=C)  # [B, C, H, W]
            gabor_responses.append(response)
        
        # 能量聚合: sqrt(mean(response^2)) 跨方向/频率, 不会正负抵消
        gabor_stack = torch.stack(gabor_responses, dim=0)  # [num_filters, B, C, H, W]
        gabor_out = torch.sqrt((gabor_stack ** 2).mean(dim=0) + 1e-6)  # [B, C, H, W]
        
        # 投影 Gabor 输出到 out_channels (使用 __init__ 中构建的可学习投影)
        gabor_out = self.gabor_proj(gabor_out)  # [B, out_channels, H, W]
        
        # 强制对齐到 conv_out 的空间尺寸: stride>1 时 avg_pool 的取整规则可能与
        # conv 的输出尺寸不一致 (尤其奇数分辨率), 直接 interpolate 保证严格匹配
        if gabor_out.shape[-2:] != (H_out, W_out):
            gabor_out = F.interpolate(gabor_out, size=(H_out, W_out),
                                      mode='bilinear', align_corners=False)
        
        # NaN/Inf 用 nan_to_num 就地清洗, 而不是提前 return (return 会丢掉 Gabor 分支
        # 的梯度路径, 且返回值语义不一致); 保持前向图完整
        gabor_out = torch.nan_to_num(gabor_out, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 调制卷积输出 - 使用安全的范围
        mod = self.modulation(x)
        conv_out = conv_out * (1.0 + mod.clamp(-0.1, 0.1))
        
        # residual fusion: out = conv_out + proj([conv_out, gabor_out])
        # proj 的 BN 零初始化保证初始残差≈0, Gabor 贡献平滑长起, 不会随机强注入
        fused = torch.cat([conv_out, gabor_out], dim=1)  # [B, 2*out_channels, H, W]
        out = conv_out + self.projection(fused)  # [B, out_channels, H, W]
        
        # 最终清洗, 保持前向图完整
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        
        return out


class NeuroHFEncoderBranch(nn.Module):
    """
    神经启发式高频编码器分支 (策略2)
    
    从原始RGB图像直接提取高频特征，形成与显式edge分支平行的独立分支。
    模拟V1视觉皮层的早期滤波机制。
    
    拓扑对照：
    ┌─────────────────────────────────────────────────────────────┐
    │  显式 Edge 分支:  edge_map → edge_down_4 → edge_down_8    │
    │  隐式 HF 分支:    RGB_raw  → hf_down_4    → hf_down_8    │
    └─────────────────────────────────────────────────────────────┘
    
    与原ImplicitHighFreqExtractor的区别：
    - 输入：原始RGB图像（而非处理后的特征）
    - 位置：编码器前端的独立分支
    - 目标：直接从图像提取高频，与edge分支形成公平对照
    
    分辨率说明：
    - hf_4: 1/4 分辨率 (H/4, W/4)，融合到 Stage 0
    - hf_8: 1/8 分辨率 (H/8, W/8)，融合到 Stage 1
    """
    
    def __init__(self, in_channels=3, dims=[64, 128, 256], 
                 num_orientations=4, num_frequencies=2,
                 use_gabor=True, use_multiscale=False):
        super().__init__()
        self.dims = dims
        self.use_gabor = use_gabor
        
        # 高频分支：从原始图像下采样
        # 输入: [B, 3, H, W] → 输出: [B, dims[0], H/4, W/4]
        
        if use_gabor:
            # 使用可学习Gabor作为第一层（模拟V1简单细胞）
            # 第一层: H → H/2
            self.hf_conv1 = LearnableGaborConv2d(
                in_channels, dims[0] // 2,
                kernel_size=5, 
                num_orientations=num_orientations,
                num_frequencies=num_frequencies,
                stride=2,  # 2x下采样
                padding=2
            )
            # 第二层: H/2 → H/4
            self.hf_conv2 = nn.Sequential(
                nn.Conv2d(dims[0] // 2, dims[0], 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(dims[0]),
                nn.ReLU(inplace=True)
            )
        else:
            # 普通卷积版本
            self.hf_conv1 = nn.Sequential(
                nn.Conv2d(in_channels, dims[0] // 2, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(dims[0] // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(dims[0] // 2, dims[0], 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(dims[0]),
                nn.ReLU(inplace=True)
            )
            self.hf_conv2 = nn.Identity()
        
        # 第三层: H/4 → H/8
        self.hf_conv3 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[1]),
            nn.ReLU(inplace=True)
        )
        
        # 输出投影 + 自适应池化确保分辨率匹配主干
        self.proj_4 = nn.Sequential(
            nn.Conv2d(dims[0], dims[0], 1, bias=False),
            nn.BatchNorm2d(dims[0])
        )
        
        self.proj_8 = nn.Sequential(
            nn.Conv2d(dims[1], dims[1], 1, bias=False),
            nn.BatchNorm2d(dims[1])
        )
        
        # 目标分辨率（暂时不需要）
        self.target_h_4 = None
        self.target_w_4 = None
        self.target_h_8 = None
        self.target_w_8 = None
        
        # 不再做零初始化: 之前的 _zero_init 会把 Gabor/proj 全部清零,
        # 导致分支输出恒为 0、梯度信号极弱, 分支实际"躺平"无法学到高频。
        # 改用默认 (Kaiming) 初始化, 让分支从一开始就有非零高频响应,
        # 融合处再用小的 learnable scale 控制注入强度 (见 encoder 融合逻辑)。
    
    def forward(self, x_raw):
        """
        Args:
            x_raw: 原始RGB图像（已normalize）[B, 3, H, W]
        Returns:
            hf_4: 1/4分辨率高频特征 [B, dims[0], H/4, W/4]
            hf_8: 1/8分辨率高频特征 [B, dims[1], H/8, W/8]
        """
        # 下采样路径: H → H/2 → H/4
        if self.use_gabor:
            h1 = self.hf_conv1(x_raw)           # [B, C/2, H/2, W/2]
            hf_4 = self.hf_conv2(h1)            # [B, C, H/4, W/4]
        else:
            hf_4 = self.hf_conv1(x_raw)         # [B, C, H/4, W/4]
        
        hf_4 = self.proj_4(hf_4)               # [B, dims[0], H/4, W/4]
        
        # 继续下采样: H/4 → H/8
        hf_8 = self.hf_conv3(hf_4)              # [B, dims[1], H/8, W/8]
        hf_8 = self.proj_8(hf_8)                # [B, dims[1], H/8, W/8]
        
        return hf_4, hf_8


class PositionalEncodingFourier(nn.Module):
    """
    Positional encoding relying on a fourier kernel matching the one used in the
    "Attention is all of Need" paper. The implementation builds on DeTR code
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """

    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, B, H, W):
        mask = torch.zeros(B, H, W).bool().to(self.token_projection.weight.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)
        return pos


class XCA(nn.Module):
    """ Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
     sum. The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)


    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class BNGELU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-5)
        self.act = nn.GELU()

    def forward(self, x):
        output = self.bn(x)
        output = self.act(output)

        return output


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding=0, dilation=(1, 1), groups=1, bn_act=False, bias=False):
        super().__init__()

        self.bn_act = bn_act

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_act:
            self.bn_gelu = BNGELU(nOut)

    def forward(self, x):
        output = self.conv(x)

        if self.bn_act:
            output = self.bn_gelu(output)

        return output


class CDilated(nn.Module):
    """
    This class defines the dilated convolution.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1, bias=False):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=bias,
                              dilation=d, groups=groups)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """

        output = self.conv(input)
        return output


class DilatedConv(nn.Module):
    """
    A single Dilated Convolution layer in the Consecutive Dilated Convolutions (CDC) module.
    """
    def __init__(self, dim, k, dilation=1, stride=1, drop_path=0.,
                 layer_scale_init_value=1e-6, expan_ratio=6):
        """
        :param dim: input dimension
        :param k: kernel size
        :param dilation: dilation rate
        :param drop_path: drop_path rate
        :param layer_scale_init_value:
        :param expan_ratio: inverted bottelneck residual
        """

        super().__init__()

        self.ddwconv = CDilated(dim, dim, kSize=k, stride=stride, groups=dim, d=dilation)
        self.bn1 = nn.BatchNorm2d(dim)

        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x

        x = self.ddwconv(x)
        x = self.bn1(x)

        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)

        return x


class LGFI(nn.Module):
    """
    Local-Global Features Interaction
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=6,
                 use_pos_emb=True, num_heads=6, qkv_bias=True, attn_drop=0., drop=0.):
        super().__init__()

        self.dim = dim
        self.pos_embd = None
        if use_pos_emb:
            self.pos_embd = PositionalEncodingFourier(dim=self.dim)

        self.norm_xca = LayerNorm(self.dim, eps=1e-6)

        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        self.xca = XCA(self.dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm = LayerNorm(self.dim, eps=1e-6)
        self.pwconv1 = nn.Linear(self.dim, expan_ratio * self.dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * self.dim, self.dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((self.dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input_ = x

        # XCA
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)

        if self.pos_embd:
            pos_encoding = self.pos_embd(B, H, W).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding

        x = x + self.gamma_xca * self.xca(self.norm_xca(x))

        x = x.reshape(B, H, W, C)

        # Inverted Bottleneck
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input_ + self.drop_path(x)

        return x


class AvgPool(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        for pool in self.pool:
            x = pool(x)

        return x


class LiteMono(nn.Module):
    """
    Lite-Mono
    """
    def __init__(self, in_chans=3, model='lite-mono', height=192, width=640,
                 global_block=[1, 1, 1], global_block_type=['LGFI', 'LGFI', 'LGFI'],
                 drop_path_rate=0.2, layer_scale_init_value=1e-6, expan_ratio=6,
                 heads=[8, 8, 8], use_pos_embd_xca=[True, False, False],
                 use_edge_mid_fusion=False, edge_fusion_strength=1.0,
                 edge_fusion_strength_mode="gate_only", edge_fusion_mode="v1",
                 edge_stage_mode="all", 
                 # Neuro-inspired implicit high-frequency fusion options
                 use_neuro_fusion=False, neuro_fusion_mode="gabor",
                 neuro_fusion_stage="all", neuro_multiscale=True,
                 **kwargs):

        super().__init__()
        self.in_chans = in_chans
        self.use_edge_mid_fusion = use_edge_mid_fusion
        self.edge_fusion_strength = edge_fusion_strength
        self.edge_fusion_strength_mode = edge_fusion_strength_mode
        self.edge_stage_mode = edge_stage_mode
        
        # Neuro-inspired fusion parameters
        self.use_neuro_fusion = use_neuro_fusion
        self.neuro_fusion_mode = neuro_fusion_mode
        self.neuro_fusion_stage = neuro_fusion_stage
        self.neuro_multiscale = neuro_multiscale

        if model == 'lite-mono':
            self.num_ch_enc = np.array([48, 80, 128])
            self.depth = [4, 4, 10]
            self.dims = [48, 80, 128]
            if height == 192 and width == 640:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]
            elif height == 320 and width == 1024:
                self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 1, 2, 5, 2, 4, 10]]

        elif model == 'lite-mono-small':
            self.num_ch_enc = np.array([48, 80, 128])
            self.depth = [4, 4, 7]
            self.dims = [48, 80, 128]
            if height == 192 and width == 640:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 2, 4, 6]]
            elif height == 320 and width == 1024:
                self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 2, 4, 10]]

        elif model == 'lite-mono-tiny':
            self.num_ch_enc = np.array([32, 64, 128])
            self.depth = [4, 4, 7]
            self.dims = [32, 64, 128]
            if height == 192 and width == 640:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 2, 4, 6]]
            elif height == 320 and width == 1024:
                self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 2, 4, 10]]

        elif model == 'lite-mono-8m':
            self.num_ch_enc = np.array([64, 128, 224])
            self.depth = [4, 4, 10]
            self.dims = [64, 128, 224]
            if height == 192 and width == 640:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]
            elif height == 320 and width == 1024:
                self.dilation = [[1, 2, 3], [1, 2, 3], [1, 2, 3, 1, 2, 3, 2, 4, 6]]

        for g in global_block_type:
            assert g in ['None', 'LGFI']

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem1 = nn.Sequential(
            Conv(in_chans, self.dims[0], kSize=3, stride=2, padding=1, bn_act=True),
            Conv(self.dims[0], self.dims[0], kSize=3, stride=1, padding=1, bn_act=True),
            Conv(self.dims[0], self.dims[0], kSize=3, stride=1, padding=1, bn_act=True),
        )

        self.stem2 = nn.Sequential(
            Conv(self.dims[0] + self.in_chans, self.dims[0], kSize=3, stride=2, padding=1, bn_act=False),
        )

        self.downsample_layers.append(stem1)

        self.input_downsample = nn.ModuleList()
        for i in range(1, 5):
            self.input_downsample.append(AvgPool(i))

        for i in range(2):
            downsample_layer = nn.Sequential(
                Conv(self.dims[i] * 2 + self.in_chans, self.dims[i + 1], kSize=3, stride=2, padding=1, bn_act=False),
            )
            self.downsample_layers.append(downsample_layer)

        self.edge_fusion_mode = edge_fusion_mode

        if self.use_edge_mid_fusion:
            # Edge branch for v1/fpn: edge -> 1/4 and 1/8 features
            edge_mid_ch = max(8, self.dims[0] // 2)

            if edge_fusion_mode == "v1":
                # Original v1: 1/4 and 1/8 fusion
                self.edge_down_4 = nn.Sequential(
                    Conv(1, edge_mid_ch, kSize=3, stride=2, padding=1, bn_act=True),
                    Conv(edge_mid_ch, self.dims[0], kSize=3, stride=2, padding=1, bn_act=True),
                )
                self.edge_down_8 = Conv(self.dims[0], self.dims[1], kSize=3, stride=2, padding=1, bn_act=True)
                # For stage ablation: 1/16 edge feature
                self.edge_down_16 = Conv(self.dims[1], self.dims[2], kSize=3, stride=2, padding=1, bn_act=True)

                self.gate_4 = nn.Sequential(
                    nn.Conv2d(self.dims[0] * 2, self.dims[0], kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.dims[0], self.dims[0], kernel_size=1),
                    nn.Sigmoid(),
                )
                self.gate_8 = nn.Sequential(
                    nn.Conv2d(self.dims[1] * 2, self.dims[1], kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.dims[1], self.dims[1], kernel_size=1),
                    nn.Sigmoid(),
                )
                self.gate_16 = nn.Sequential(
                    nn.Conv2d(self.dims[2] * 2, self.dims[2], kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.dims[2], self.dims[2], kernel_size=1),
                    nn.Sigmoid(),
                )

            elif edge_fusion_mode == "v2":
                # V2: Lightweight gating with reduced intermediate dimensions
                edge_mid_ch_v2 = max(6, self.dims[0] // 4)

                self.edge_down_4 = nn.Sequential(
                    Conv(1, edge_mid_ch_v2, kSize=3, stride=2, padding=1, bn_act=True),
                    Conv(edge_mid_ch_v2, self.dims[0], kSize=3, stride=2, padding=1, bn_act=True),
                )
                self.edge_down_8 = Conv(self.dims[0], self.dims[1], kSize=3, stride=2, padding=1, bn_act=True)

                self.gate_4 = nn.Sequential(
                    nn.Conv2d(self.dims[0] * 2, max(8, self.dims[0] // 8), kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(max(8, self.dims[0] // 8), self.dims[0], kernel_size=1),
                    nn.Sigmoid(),
                )
                self.gate_8 = nn.Sequential(
                    nn.Conv2d(self.dims[1] * 2, max(8, self.dims[1] // 8), kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(max(8, self.dims[1] // 8), self.dims[1], kernel_size=1),
                    nn.Sigmoid(),
                )

            elif edge_fusion_mode == "v3":
                # V3: C/4 intermediate (balanced between v1 and v2)
                edge_mid_ch_v3 = self.dims[0] // 4

                self.edge_down_4 = nn.Sequential(
                    Conv(1, edge_mid_ch_v3, kSize=3, stride=2, padding=1, bn_act=True),
                    Conv(edge_mid_ch_v3, self.dims[0], kSize=3, stride=2, padding=1, bn_act=True),
                )
                self.edge_down_8 = Conv(self.dims[0], self.dims[1], kSize=3, stride=2, padding=1, bn_act=True)

                self.gate_4 = nn.Sequential(
                    nn.Conv2d(self.dims[0] * 2, self.dims[0] // 4, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.dims[0] // 4, self.dims[0], kernel_size=1),
                    nn.Sigmoid(),
                )
                self.gate_8 = nn.Sequential(
                    nn.Conv2d(self.dims[1] * 2, self.dims[1] // 4, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.dims[1] // 4, self.dims[1], kernel_size=1),
                    nn.Sigmoid(),
                )

            elif edge_fusion_mode == "v4":
                # V4: C/2 intermediate (larger than v1)
                edge_mid_ch_v4 = self.dims[0] // 2

                self.edge_down_4 = nn.Sequential(
                    Conv(1, edge_mid_ch_v4, kSize=3, stride=2, padding=1, bn_act=True),
                    Conv(edge_mid_ch_v4, self.dims[0], kSize=3, stride=2, padding=1, bn_act=True),
                )
                self.edge_down_8 = Conv(self.dims[0], self.dims[1], kSize=3, stride=2, padding=1, bn_act=True)

                self.gate_4 = nn.Sequential(
                    nn.Conv2d(self.dims[0] * 2, self.dims[0] // 2, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.dims[0] // 2, self.dims[0], kernel_size=1),
                    nn.Sigmoid(),
                )
                self.gate_8 = nn.Sequential(
                    nn.Conv2d(self.dims[1] * 2, self.dims[1] // 2, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.dims[1] // 2, self.dims[1], kernel_size=1),
                    nn.Sigmoid(),
                )

            elif edge_fusion_mode == "film":
                # FiLM: Feature-wise Linear Modulation
                # gamma, beta = edge_gate(edge_feat)
                # x = x * (1 + alpha * gamma) + alpha * beta
                edge_mid_ch_film = max(8, self.dims[0] // 2)

                self.edge_down_4 = nn.Sequential(
                    Conv(1, edge_mid_ch_film, kSize=3, stride=2, padding=1, bn_act=True),
                    Conv(edge_mid_ch_film, self.dims[0], kSize=3, stride=2, padding=1, bn_act=True),
                )
                self.edge_down_8 = Conv(self.dims[0], self.dims[1], kSize=3, stride=2, padding=1, bn_act=True)

                # FiLM generator: outputs gamma (scale) and beta (shift)
                self.film_gate_4 = nn.Sequential(
                    nn.Conv2d(self.dims[0] * 2, self.dims[0] // 2, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.dims[0] // 2, self.dims[0] * 2, kernel_size=1),  # gamma and beta
                )
                self.film_gate_8 = nn.Sequential(
                    nn.Conv2d(self.dims[1] * 2, self.dims[1] // 2, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.dims[1] // 2, self.dims[1] * 2, kernel_size=1),  # gamma and beta
                )
        
        # =====================================================================
        # Neuro-inspired Implicit High-Frequency Feature Fusion
        # 神经启发式隐式高频特征融合
        # =====================================================================
        # =====================================================================
        # Neuro-inspired implicit high-frequency encoder branch (策略2)
        # 
        # 从原始RGB图像直接提取高频特征，形成与显式edge分支平行的独立分支。
        # 
        # 拓扑对照：
        #   显式 Edge 分支:  edge_map → edge_down_4 → edge_down_8
        #   隐式 HF 分支:    RGB_raw  → hf_down_4    → hf_down_8
        #
        # 这样修改后更符合：
        # - V1早期视觉系统的Gabor滤波机制
        # - 与显式Sobel/Canny边缘提取形成公平对照实验
        # =====================================================================
        if self.use_neuro_fusion:
            # 新的 Neuro HF Encoder 分支：从原始RGB直接提取高频
            self.neuro_hf_branch = NeuroHFEncoderBranch(
                in_channels=3,  # RGB图像
                dims=self.dims,
                num_orientations=4,
                num_frequencies=2,
                use_gabor=True,
                use_multiscale=neuro_multiscale
            )
            
            # 注意：不再需要从特征图提取高频，所以下面的模块暂时保留用于兼容
            # 但实际使用时直接用 neuro_hf_branch 的输出
            
            # Fusion gates for neuro features
            self.neuro_gate_4 = nn.Sequential(
                nn.Conv2d(self.dims[0] * 2, self.dims[0] // 2, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.dims[0] // 2, self.dims[0], kernel_size=1),
                nn.Sigmoid()
            )
            # 策略2：使用 neuro_hf_branch 从 RGB 提取高频
            self.neuro_gate_8 = nn.Sequential(
                nn.Conv2d(self.dims[1] * 2, self.dims[1] // 2, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.dims[1] // 2, self.dims[1], kernel_size=1),
                nn.Sigmoid()
            )
            # 融合注入强度: 用小的 learnable scale 控制高频分支贡献,
            # 初始 0.1 让分支起步时对主干扰动小, 但梯度可让其自适应增大,
            # 取代之前"把分支初始化成 0"的做法。
            self.neuro_scale_4 = nn.Parameter(torch.tensor(0.1))
            self.neuro_scale_8 = nn.Parameter(torch.tensor(0.1))

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depth))]
        cur = 0
        for i in range(3):
            stage_blocks = []
            for j in range(self.depth[i]):
                if j > self.depth[i] - global_block[i] - 1:
                    if global_block_type[i] == 'LGFI':
                        stage_blocks.append(LGFI(dim=self.dims[i], drop_path=dp_rates[cur + j],
                                                 expan_ratio=expan_ratio,
                                                 use_pos_emb=use_pos_embd_xca[i], num_heads=heads[i],
                                                 layer_scale_init_value=layer_scale_init_value,
                                                 ))

                    else:
                        raise NotImplementedError
                else:
                    stage_blocks.append(DilatedConv(dim=self.dims[i], k=3, dilation=self.dilation[i][j], drop_path=dp_rates[cur + j],
                                                    layer_scale_init_value=layer_scale_init_value,
                                                    expan_ratio=expan_ratio))

            self.stages.append(nn.Sequential(*stage_blocks))
            cur += self.depth[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def _fuse_neuro(self, x, neuro_module, gate_module):
        """Neuro-inspired fusion: learned high-frequency features with gating"""
        # Extract implicit high-frequency features
        neuro_feat = neuro_module(x)
        # Gate the fusion
        gate = gate_module(torch.cat([x, neuro_feat], dim=1))
        fused = neuro_feat * gate
        return x + fused
    
    def _fuse_edge(self, x, edge_feat, gate_module):
        """Original v1 fusion: element-wise multiplication"""
        gate = gate_module(torch.cat([x, edge_feat], dim=1))
        fused = edge_feat * gate
        return x + fused

    def _fuse_edge_v2(self, x, edge_feat, gate_module):
        """V2 fusion: residual connection with gated edge contribution"""
        gate = gate_module(torch.cat([x, edge_feat], dim=1))
        return x + edge_feat * gate

    def _fuse_edge_film(self, x, edge_feat, film_gate):
        """FiLM fusion: Feature-wise Linear Modulation
        gamma, beta = film_gate(concat(x, edge_feat))
        x = x * (1 + gamma) + beta
        """
        cond = film_gate(torch.cat([x, edge_feat], dim=1))
        gamma, beta = torch.chunk(cond, 2, dim=1)
        gamma = torch.tanh(gamma)  # bounded scale
        return x * (1 + gamma) + beta

    def forward_features(self, x, edge=None):
        features = []
        x = (x - 0.45) / 0.225
        
        # 保存原始RGB图像用于Neuro HF分支（策略2：从源图像提取高频）
        x_raw = x.clone()
        
        # Prepare edge features
        edge_4 = None
        edge_8 = None
        edge_16 = None
        if self.use_edge_mid_fusion and edge is not None:
            edge_4 = self.edge_down_4(edge)
            edge_8 = self.edge_down_8(edge_4)
            if self.edge_stage_mode == "stage0_stage1" or self.edge_stage_mode == "all":
                if hasattr(self, 'edge_down_16'):
                    edge_16 = self.edge_down_16(edge_8)
        
        # 策略2：Neuro HF分支直接从原始RGB提取高频特征
        hf_4 = None
        hf_8 = None
        if self.use_neuro_fusion and hasattr(self, 'neuro_hf_branch'):
            hf_4, hf_8 = self.neuro_hf_branch(x_raw)
        
        x_down = []
        for i in range(4):
            x_down.append(self.input_downsample[i](x))

        tmp_x = []
        x = self.downsample_layers[0](x)
        x = self.stem2(torch.cat((x, x_down[0]), dim=1))

        tmp_x.append(x)

        for s in range(len(self.stages[0])-1):
            x = self.stages[0][s](x)
        x = self.stages[0][-1](x)

        # Stage 0 (1/4 resolution) - edge or neuro fusion
        if edge_4 is not None and (self.edge_stage_mode == "stage0_only" or self.edge_stage_mode == "all"):
            if self.edge_fusion_mode == "film":
                x = self._fuse_edge_film(x, edge_4, self.film_gate_4)
            elif self.edge_fusion_mode == "v2":
                x = self._fuse_edge_v2(x, edge_4, self.gate_4)
            else:
                x = self._fuse_edge(x, edge_4, self.gate_4)
        
        # Neuro HF分支融合（策略2：从源图像提取的高频特征）
        if self.use_neuro_fusion and hf_4 is not None:
            # 使用 neuro_hf_branch 输出的 hf_4 与主干 x 融合
            gate = self.neuro_gate_4(torch.cat([x, hf_4], dim=1))
            x = x + self.neuro_scale_4 * hf_4 * gate

        tmp_x.append(x)
        features.append(x)

        for i in range(1, 3):
            tmp_x.append(x_down[i])
            x = torch.cat(tmp_x, dim=1)
            x = self.downsample_layers[i](x)

            tmp_x = [x]
            for s in range(len(self.stages[i]) - 1):
                x = self.stages[i][s](x)
            x = self.stages[i][-1](x)

            # Stage 1 (1/8 resolution) - edge or neuro fusion
            if i == 1 and edge_8 is not None and (self.edge_stage_mode == "stage1_only" or self.edge_stage_mode == "all"):
                if self.edge_fusion_mode == "film":
                    x = self._fuse_edge_film(x, edge_8, self.film_gate_8)
                elif self.edge_fusion_mode == "v2":
                    x = self._fuse_edge_v2(x, edge_8, self.gate_8)
                else:
                    x = self._fuse_edge(x, edge_8, self.gate_8)
            
            # Neuro HF分支融合（策略2：使用从源图像提取的hf_8特征）
            if i == 1 and self.use_neuro_fusion and hf_8 is not None:
                gate = self.neuro_gate_8(torch.cat([x, hf_8], dim=1))
                x = x + self.neuro_scale_8 * hf_8 * gate

            # Stage 2 (1/16 resolution) - edge or neuro fusion
            elif i == 2 and edge_16 is not None:
                if self.edge_stage_mode == "stage0_stage1":
                    x = self._fuse_edge(x, edge_16, self.gate_16)
            
            # 注意：Stage 2 不再使用neuro fusion，因为hf_8下采样后已经是1/8
            # 如果需要1/16的高频，可以从hf_8继续下采样得到

            tmp_x.append(x)
            features.append(x)

        return features

    def forward(self, x, edge=None):
        x = self.forward_features(x, edge=edge)
        return x
