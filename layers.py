from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class ConvBlockDepth(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlockDepth, self).__init__()

        self.conv = DepthConv3x3(in_channels, out_channels)
        self.nonlin = nn.GELU()

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class DepthConv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(DepthConv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        # self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=3, groups=int(out_channels), bias=False)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)
        # self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=3, padding=3 // 2, groups=int(out_channels), bias=False)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


def upsample(x, scale_factor=2, mode="bilinear"):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode)


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


# =============================================================================
# Sampling Strategy Functions (深度与姿态联合采样策略)
# =============================================================================

def depth_sampling(disp, num_samples, noise_std, mode="scaled_disp", min_depth=0.1, max_depth=100.0):
    """
    在深度空间进行相对扰动采样
    
    Args:
        disp: 原始 sigmoid 输出 [B, 1, H, W], 范围 [0, 1]
        num_samples: 采样数量
        noise_std: 噪声标准差（用于 log 或 ratio 空间）
        mode: "scaled_disp" / "log_depth" / "disp"
        min_depth: 最小深度 (默认 0.1m)
        max_depth: 最大深度 (默认 100.0m)
    
    Returns:
        disp_samples: [num_samples, B, 1, H, W] 的候选视差列表
    """
    # 确保 disp 是 [B, 1, H, W] 格式
    if disp.dim() == 3:  # [B, H, W] -> [B, 1, H, W]
        disp = disp.unsqueeze(1)
    
    B = disp.shape[0]
    min_disp = 1.0 / max_depth   # 0.01
    max_disp = 1.0 / min_depth   # 10.0
    
    if mode == "scaled_disp":
        # 在 scaled_disp 上做相对扰动: scaled_disp * exp(eps)
        scaled_disp = min_disp + (max_disp - min_disp) * disp  # [B, 1, H, W]
        log_scaled = torch.log(scaled_disp + 1e-8)
        
        # 生成噪声并扩展维度 - [num_samples, B, 1, 1, 1]
        noise = torch.randn(num_samples, B, 1, 1, 1, device=disp.device, dtype=disp.dtype) * noise_std
        log_scaled_expanded = log_scaled.unsqueeze(0).expand(num_samples, -1, -1, -1, -1)
        log_scaled_samples = log_scaled_expanded + noise
        scaled_disp_samples = torch.exp(log_scaled_samples)
        disp_samples = (scaled_disp_samples - min_disp) / (max_disp - min_disp)
        disp_samples = torch.clamp(disp_samples, 0, 1)
        
    elif mode == "log_depth":
        # 在 log depth 上做加性扰动
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1.0 / (scaled_disp + 1e-8)
        log_depth = torch.log(depth + 1e-8)
        
        noise = torch.randn(num_samples, B, 1, 1, 1, device=disp.device, dtype=disp.dtype) * noise_std
        log_depth_expanded = log_depth.unsqueeze(0).expand(num_samples, -1, -1, -1, -1)
        log_depth_samples = log_depth_expanded + noise
        depth_samples = torch.exp(log_depth_samples)
        scaled_disp_samples = 1.0 / (depth_samples + 1e-8)
        disp_samples = (scaled_disp_samples - min_disp) / (max_disp - min_disp)
        disp_samples = torch.clamp(disp_samples, 0, 1)
        
    else:  # "disp"
        # 直接在 disp 上做相对扰动: disp * exp(eps)
        log_disp = torch.log(disp + 1e-8)
        noise = torch.randn(num_samples, B, 1, 1, 1, device=disp.device, dtype=disp.dtype) * noise_std
        log_disp_expanded = log_disp.unsqueeze(0).expand(num_samples, -1, -1, -1, -1)
        log_disp_samples = log_disp_expanded + noise
        disp_samples = torch.exp(log_disp_samples)
        disp_samples = torch.clamp(disp_samples, 0, 1)
    
    return disp_samples  # [N, B, 1, H, W]


def pose_sampling(axisangle, translation, num_samples, rot_noise, trans_noise_ratio):
    """
    对姿态进行采样
    
    Args:
        axisangle: [B, N, 1, 3] 原始旋转输出
        translation: [B, N, 1, 3] 原始平移输出
        num_samples: 采样数量
        rot_noise: 旋转噪声 (rad)，加性高斯噪声在弧度空间
        trans_noise_ratio: 平移相对扰动比例
    
    Returns:
        axisangle_samples: [num_samples, B, N, 1, 3]
        translation_samples: [num_samples, B, N, 1, 3]
    """
    B = axisangle.shape[0]
    N = axisangle.shape[1]
    
    # 旋转：加性高斯噪声（弧度空间）
    rot_eps = torch.randn(num_samples, 1, 1, 1, 3, device=axisangle.device, dtype=axisangle.dtype) * rot_noise
    axisangle_expanded = axisangle.unsqueeze(0).expand(num_samples, -1, -1, -1, -1)
    axisangle_samples = axisangle_expanded + rot_eps
    
    # 平移：相对扰动 t * (1 + eps)
    trans_eps = torch.randn(num_samples, 1, 1, 1, 3, device=translation.device, dtype=translation.dtype) * trans_noise_ratio
    trans_expanded = translation.unsqueeze(0).expand(num_samples, -1, -1, -1, -1)
    translation_samples = trans_expanded * (1.0 + trans_eps)
    
    return axisangle_samples, translation_samples


def pose_constraint_loss(axisangle_best, axisangle_hat, translation_best, translation_hat):
    """
    在 pose vector 空间计算一致性约束（不在 4x4 矩阵上做差）
    
    Args:
        axisangle_best: [B, N, 1, 3] 最优采样旋转
        axisangle_hat: [B, N, 1, 3] 预测旋转
        translation_best: [B, N, 1, 3] 最优采样平移
        translation_hat: [B, N, 1, 3] 预测平移
    
    Returns:
        rot_loss: ||axisangle_best - axisangle_hat||_1
        trans_loss: ||translation_best - translation_hat||_1
    """
    rot_loss = torch.abs(axisangle_best - axisangle_hat).mean()
    trans_loss = torch.abs(translation_best - translation_hat).mean()
    return rot_loss, trans_loss


