"""
可视化 lite-mono-sampling-dual 模型的深度预测结果 - 包含残差分析
"""
import os
import sys
import numpy as np
import cv2
import torch
from pathlib import Path

sys.path.insert(0, '/home/jzy/data/vision/my-mono')
from layers import disp_to_depth

def depth_to_colormap(depth, min_d=None, max_d=None):
    """将深度转换为彩色可视化"""
    if min_d is None:
        min_d = depth.min()
    if max_d is None:
        max_d = depth.max()
    depth_norm = (depth - min_d) / (max_d - min_d + 1e-8)
    depth_norm = np.clip(depth_norm, 0, 1)
    return cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

def residual_to_colormap(residual):
    """残差转彩色 - 蓝0红正 白负"""
    residual = residual.copy()
    residual = np.clip(residual, -3, 3)
    residual_norm = (residual + 3) / 6.0
    residual_norm = np.clip(residual_norm, 0, 1)
    cmap = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        if t < 0.5:
            cmap[i] = [255 * (1 - 2*t), 255 * (1 - 2*t), 255]
        else:
            cmap[i] = [255, 255 * (2 - 2*t), 255 * (2 - 2*t)]
    idx = (residual_norm * 255).astype(np.uint8)
    return cmap[idx]

def main():
    data_path = Path('/home/jzy/data/vision/monodepth2/kitti_data')
    output_dir = Path('/home/jzy/data/vision/my-mono/vis_results/sampling_dual')
    output_dir.mkdir(exist_ok=True)
    
    # 加载预测的视差
    disp_path = Path('/home/jzy/data/vision/my-mono/tmp/lite-mono-sampling-dual/models/weights_49/disps_eigen_split.npy')
    disps = np.load(disp_path)
    
    # 加载GT深度
    gt_path = Path('/home/jzy/data/vision/my-mono/splits/eigen/gt_depths.npz')
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
    print(f"Loaded {len(gt_depths)} GT depths")
    
    # KITTI eigen test split 场景 (与评测时一致)
    test_frames = {
        0: ('2011_09_26/2011_09_26_drive_0011_sync', 14),
        1: ('2011_09_26/2011_09_26_drive_0011_sync', 20),
        2: ('2011_09_26/2011_09_26_drive_0011_sync', 30),
        3: ('2011_09_26/2011_09_26_drive_0009_sync', 100),
        4: ('2011_09_26/2011_09_26_drive_0009_sync', 120),
        5: ('2011_09_26/2011_09_26_drive_0009_sync', 140),
        6: ('2011_09_30/2011_09_30_drive_0016_sync', 50),
        7: ('2011_09_30/2011_09_30_drive_0016_sync', 80),
        8: ('2011_09_30/2011_09_30_drive_0018_sync', 30),
        9: ('2011_09_30/2011_09_30_drive_0018_sync', 60),
    }
    
    MIN_DEPTH, MAX_DEPTH = 0.1, 100.0
    
    for idx, (seq, frame) in test_frames.items():
        if idx >= len(disps) or idx >= len(gt_depths):
            continue
        
        seq_path = data_path / seq
        img_path = seq_path / 'image_02' / 'data' / f'{frame:010d}.png'
        
        if not img_path.exists():
            print(f"Image not found: {img_path}")
            continue
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        h_orig, w_orig = img.shape[:2]
        gt_depth = gt_depths[idx]
        gt_height, gt_width = gt_depth.shape[:2]
        
        # 预测视差
        disp = disps[idx]
        pred_disp = cv2.resize(disp, (gt_width, gt_height))
        pred_depth_full = 1.0 / (pred_disp + 1e-8)
        
        # 应用median scaling (与评测一致)
        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        pred_depth_crop = pred_depth_full[mask]
        gt_depth_crop = gt_depth[mask]
        
        ratio = np.median(gt_depth_crop) / np.median(pred_depth_crop)
        pred_depth_scaled = pred_depth_full * ratio
        
        # 裁剪区域
        crop = np.array([
            0.40810811 * gt_height, 0.99189189 * gt_height,
            0.03594771 * gt_width,  0.96405229 * gt_width
        ]).astype(np.int32)
        crop_mask = np.zeros(mask.shape, dtype=bool)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = True
        mask = np.logical_and(mask, crop_mask)
        
        # 缩放到原始尺寸
        disp_resized = cv2.resize(disp, (w_orig, h_orig))
        depth_resized = cv2.resize(pred_depth_scaled, (w_orig, h_orig))
        gt_resized = cv2.resize(gt_depth, (w_orig, h_orig))
        
        # 计算残差
        residual = (pred_depth_scaled - gt_depth) / (gt_depth + 1e-8)
        
        prefix = f"{idx:02d}_{seq.replace('/', '_')}_{frame}"
        
        # 可视化
        rgb_vis = img
        disp_color = depth_to_colormap(disp_resized)
        depth_color = depth_to_colormap(depth_resized)
        gt_color = depth_to_colormap(gt_resized)
        residual_color = residual_to_colormap(residual)
        residual_resized = cv2.resize(residual_color, (w_orig, h_orig))
        
        # 拼接
        target_h = 300
        scale = target_h / h_orig
        target_w = int(w_orig * scale)
        
        def resize_keep_aspect(img, h):
            s = h / img.shape[0]
            w = int(img.shape[1] * s)
            return cv2.resize(img, (w, h))
        
        def add_label(img, text):
            h, w = img.shape[:2]
            label = np.zeros((45, w, 3), dtype=np.uint8) + 25
            cv2.putText(label, text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
            return np.vstack([label, img])
        
        rgb_small = add_label(resize_keep_aspect(rgb_vis, target_h), 'RGB Input')
        disp_small = add_label(resize_keep_aspect(disp_color, target_h), 'Pred Disparity')
        depth_small = add_label(resize_keep_aspect(depth_color, target_h), 'Pred Depth')
        gt_small = add_label(resize_keep_aspect(gt_color, target_h), 'GT Depth')
        res_small = add_label(resize_keep_aspect(residual_resized, target_h), 'Error (blue=under, red=over)')
        
        # 第一行: RGB + PredDisp + PredDepth
        row1 = np.hstack([rgb_small, disp_small, depth_small])
        # 第二行: GT Depth + Error
        row2 = np.hstack([gt_small, res_small, np.zeros_like(gt_small)])
        
        combined = np.vstack([row1, row2])
        
        cv2.imwrite(str(output_dir / f"{prefix}_combined.png"), combined)
        
        # 统计
        valid_mask = mask & (gt_depth > MIN_DEPTH) & (gt_depth < MAX_DEPTH)
        rel_err = np.abs(residual[valid_mask])
        abs_rel = np.mean(rel_err)
        print(f"{prefix}: abs_rel={abs_rel:.3f}, pred_mean={pred_depth_scaled[valid_mask].mean():.1f}m, gt_mean={gt_depth[valid_mask].mean():.1f}m")
    
    print(f"\nAll saved to: {output_dir}")

if __name__ == '__main__':
    main()
