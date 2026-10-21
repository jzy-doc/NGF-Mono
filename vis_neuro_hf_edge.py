"""
可视化 neuro_fusion 模块(Gabor)输出的高频信息与边界的关系
"""
import os
import sys
sys.path.insert(0, '/home/jzy/data/vision/my-mono')

import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from networks.depth_encoder import LiteMono


def load_encoder():
    """加载 neuro_fusion encoder"""
    encoder = LiteMono(
        model='lite-mono',
        num_layers=18,
        pretrained=False,
        height=192,
        width=640,
        use_neuro_fusion=True,
        neuro_fusion_mode="gabor",
        neuro_fusion_stage="all",
        neuro_multiscale=True,
    )
    
    weights_path = "/home/jzy/data/vision/my-mono/tmp/neuro_fusion/models/weights_29"
    state_dict = torch.load(os.path.join(weights_path, "encoder.pth"), map_location='cpu')
    # 过滤掉额外的key
    filtered_dict = {k: v for k, v in state_dict.items() 
                     if k not in ['height', 'width', 'use_stereo']}
    encoder.load_state_dict(filtered_dict)
    encoder.cuda()
    encoder.eval()
    return encoder


def extract_neuro_hf(encoder, img_tensor):
    """提取 neuro_hf_branch 的高频特征"""
    with torch.no_grad():
        x_raw = img_tensor.cuda()
        
        if hasattr(encoder, 'neuro_hf_branch'):
            hf_4, hf_8 = encoder.neuro_hf_branch(x_raw)
            return hf_4, hf_8
        return None, None


def main():
    # 加载 encoder
    print("Loading encoder...")
    encoder = load_encoder()
    
    # 获取 neuro_hf_branch
    neuro_hf = encoder.neuro_hf_branch
    print(f"neuro_hf_branch: {neuro_hf}")
    
    # 读取测试图像
    kitti_path = "/home/jzy/data/vision/monodepth2/kitti_data"
    # 查找任一drive的数据
    import glob
    img_dirs = glob.glob(os.path.join(kitti_path, "*/*/image_02/data"))
    if img_dirs:
        img_dir = img_dirs[0]
    else:
        print(f"No image data found in {kitti_path}")
        return
    
    import random
    random.seed(42)
    all_imgs = sorted([f for f in os.listdir(img_dir) if f.endswith('.png') or f.endswith('.jpg')])
    selected = random.sample(all_imgs, min(5, len(all_imgs)))
    
    results = []
    output_dir = "/home/jzy/data/vision/my-mono/vis_results/neuro_hf_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, img_name in enumerate(selected):
        print(f"\nProcessing {idx+1}/{len(selected)}: {img_name}")
        
        # 读取图像
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 提取Canny边缘
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(img_gray, 50, 150)
        
        # 预处理图像
        img_resized = cv2.resize(img, (640, 192))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # [1, 3, H, W]
        
        # 提取高频特征
        hf_4, hf_8 = extract_neuro_hf(encoder, img_tensor)
        
        if hf_4 is None:
            print(f"  No HF features!")
            continue
        
        # hf_4: [B, C, H, W] - 1/4分辨率
        hf_4_np = hf_4[0].cpu().numpy() if isinstance(hf_4, tuple) else hf_4.cpu().numpy()
        hf_8_np = hf_8[0].cpu().numpy() if isinstance(hf_8, tuple) else hf_8.cpu().numpy()
        
        print(f"  hf_4 shape: {hf_4_np.shape}, hf_8 shape: {hf_8_np.shape}")
        
        # PyTorch格式: [B, C, H, W]
        # 取第一个样本，取通道均值作为响应图
        hf_4_np = hf_4_np[0]  # [C, H, W] = [48, 48, 160]
        hf_8_np = hf_8_np[0]  # [C, H, W] = [80, 24, 80]
        
        hf_4_mean = np.mean(hf_4_np, axis=0)  # [H/4, W/4] = [48, 160]
        hf_8_mean = np.mean(hf_8_np, axis=0)  # [H/8, W/8] = [24, 80]
        
        print(f"  hf_4_mean shape: {hf_4_mean.shape}, hf_8_mean shape: {hf_8_mean.shape}")
        print(f"  edge shape: {edges.shape}")
        
        # 上采样到原始尺寸
        h, w = edges.shape
        hf_4_up = cv2.resize(hf_4_mean, (w, h))
        hf_8_up = cv2.resize(hf_8_mean, (w, h))
        
        # 归一化
        hf_4_norm = (hf_4_up - hf_4_up.min()) / (hf_4_up.max() - hf_4_up.min() + 1e-8)
        hf_8_norm = (hf_8_up - hf_8_up.min()) / (hf_8_up.max() - hf_8_up.min() + 1e-8)
        
        # 边缘二值化
        edge_binary = (edges > 0).astype(np.float32)
        
        # 计算相关性
        corr_4 = np.corrcoef(hf_4_norm.flatten(), edge_binary.flatten())[0, 1]
        corr_8 = np.corrcoef(hf_8_norm.flatten(), edge_binary.flatten())[0, 1]
        
        # 阈值化高频响应 (top 10%)
        thresh_4 = np.percentile(hf_4_norm, 90)
        thresh_8 = np.percentile(hf_8_norm, 90)
        hf_4_top = (hf_4_norm > thresh_4).astype(np.float32)
        hf_8_top = (hf_8_norm > thresh_8).astype(np.float32)
        
        # 计算与边缘的重合率
        overlap_4 = np.sum(hf_4_top * edge_binary) / (np.sum(hf_4_top) + 1e-8)
        overlap_8 = np.sum(hf_8_top * edge_binary) / (np.sum(hf_8_top) + 1e-8)
        
        print(f"  corr_4={corr_4:.3f}, corr_8={corr_8:.3f}")
        print(f"  overlap_4={overlap_4*100:.1f}%, overlap_8={overlap_8*100:.1f}%")
        
        results.append({
            'img': img_name,
            'corr_4': corr_4,
            'corr_8': corr_8,
            'overlap_4': overlap_4,
            'overlap_8': overlap_8
        })
        
        # 可视化
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        axes[0, 0].imshow(img_resized)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(edges, cmap='gray')
        axes[0, 1].set_title('Canny Edge')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(hf_4_norm, cmap='hot')
        axes[0, 2].set_title(f'hf_4 Response (1/4 scale)\ncorr={corr_4:.3f}')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(hf_8_norm, cmap='hot')
        axes[0, 3].set_title(f'hf_8 Response (1/8 scale)\ncorr={corr_8:.3f}')
        axes[0, 3].axis('off')
        
        # 第二行：热力图叠加与统计分析
        # 叠加图
        overlay = np.zeros((*edges.shape, 3))
        overlay[..., 0] = edge_binary * 0.7  # 红色边缘
        overlay[..., 1] = hf_4_norm * 0.5    # 绿色高频
        axes[1, 0].imshow(np.clip(overlay, 0, 1))
        axes[1, 0].set_title('Overlay (R=Edge, G=HF_4)')
        axes[1, 0].axis('off')
        
        # Top 10% 高频区域
        top_overlay = np.zeros((*edges.shape, 3))
        top_overlay[..., 0] = edge_binary * 0.5
        top_overlay[..., 1] = hf_4_top * 0.8
        axes[1, 1].imshow(np.clip(top_overlay, 0, 1))
        axes[1, 1].set_title(f'Top 10% HF on Edge\n(overlap={overlap_4*100:.1f}%)')
        axes[1, 1].axis('off')
        
        # 柱状图
        axes[1, 2].bar(['hf_4', 'hf_8'], [corr_4, corr_8], color=['steelblue', 'darkorange'])
        axes[1, 2].set_ylabel('Correlation')
        axes[1, 2].set_title('Correlation with Canny Edge')
        axes[1, 2].set_ylim(-0.1, 1.0)
        axes[1, 2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        axes[1, 3].bar(['hf_4', 'hf_8'], [overlap_4*100, overlap_8*100], color=['steelblue', 'darkorange'])
        axes[1, 3].set_ylabel('Overlap %')
        axes[1, 3].set_title('Top 10% HF on Edge Region')
        axes[1, 3].set_ylim(0, 100)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"hf_edge_{idx:03d}_{img_name[:-4]}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")
    
    # 汇总
    if results:
        print("\n" + "="*60)
        print("Summary:")
        print(f"  Avg correlation hf_4: {np.mean([r['corr_4'] for r in results]):.3f}")
        print(f"  Avg correlation hf_8: {np.mean([r['corr_8'] for r in results]):.3f}")
        print(f"  Avg overlap hf_4: {np.mean([r['overlap_4'] for r in results])*100:.1f}%")
        print(f"  Avg overlap hf_8: {np.mean([r['overlap_8'] for r in results])*100:.1f}%")
        print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
