"""
生成采样方式对比可视化 - 三种采样方式对比
"""
import os
import sys
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, '/home/jzy/data/vision/my-mono')

def depth_to_colormap(depth, min_d=None, max_d=None):
    if min_d is None:
        min_d = depth.min()
    if max_d is None:
        max_d = depth.max()
    depth_norm = (depth - min_d) / (max_d - min_d + 1e-8)
    depth_norm = np.clip(depth_norm, 0, 1)
    return cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

def residual_to_colormap(residual):
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
    return cmap[(residual_norm * 255).astype(np.uint8)]

def add_label(img, text):
    h, w = img.shape[:2]
    label = np.zeros((45, w, 3), dtype=np.uint8) + 25
    cv2.putText(label, text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return np.vstack([label, img])

def resize_keep_aspect(img, h):
    s = h / img.shape[0]
    w = int(img.shape[1] * s)
    return cv2.resize(img, (w, h))

def main():
    data_path = Path('/home/jzy/data/vision/monodepth2/kitti_data')
    gt_path = Path('/home/jzy/data/vision/my-mono/splits/eigen/gt_depths.npz')
    output_dir = Path('/home/jzy/data/vision/my-mono/vis_results/comparison')
    output_dir.mkdir(exist_ok=True)
    
    # 模型列表
    models = [
        ('sampling-dual', 'tmp/lite-mono-sampling-dual/models/weights_49/disps_eigen_split.npy'),
        ('sampling-depth', 'tmp/lite-mono-sampling-depth-only/models/weights_49/disps_eigen_split.npy'),
        ('sampling-pose', 'tmp/lite-mono-sampling-pose-only/models/weights_49/disps_eigen_split.npy'),
    ]
    
    disps_dict = {}
    for name, path in models:
        disps_dict[name] = np.load(path)
    
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
    
    test_frames = [
        (0, '2011_09_26/2011_09_26_drive_0011_sync', 14),
        (1, '2011_09_26/2011_09_26_drive_0011_sync', 20),
        (2, '2011_09_26/2011_09_26_drive_0009_sync', 100),
        (3, '2011_09_26/2011_09_26_drive_0009_sync', 140),
        (4, '2011_09_30/2011_09_30_drive_0016_sync', 50),
        (5, '2011_09_30/2011_09_30_drive_0018_sync', 60),
    ]
    
    MIN_DEPTH, MAX_DEPTH = 0.1, 100.0
    target_h = 200
    
    for idx, seq, frame in test_frames:
        img_path = data_path / seq / 'image_02' / 'data' / f'{frame:010d}.png'
        if not img_path.exists():
            continue
        
        img = cv2.imread(str(img_path))
        h_orig, w_orig = img.shape[:2]
        gt_depth = gt_depths[idx]
        gt_height, gt_width = gt_depth.shape[:2]
        
        prefix = f"{idx:02d}_{seq.replace('/', '_')}_{frame}"
        
        # RGB
        rgb_small = resize_keep_aspect(img, target_h)
        
        cols = [add_label(rgb_small, 'RGB')]
        
        for name, disps in disps_dict.items():
            if idx >= len(disps):
                continue
            
            disp = disps[idx]
            pred_disp = cv2.resize(disp, (gt_width, gt_height))
            pred_depth_full = 1.0 / (pred_disp + 1e-8)
            
            # Median scaling
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            ratio = np.median(gt_depth[mask]) / np.median(pred_depth_full[mask])
            pred_depth_scaled = pred_depth_full * ratio
            
            # 缩放到原始尺寸
            depth_resized = cv2.resize(pred_depth_scaled, (w_orig, h_orig))
            depth_color = depth_to_colormap(depth_resized)
            
            # 残差
            residual = (pred_depth_scaled - gt_depth) / (gt_depth + 1e-8)
            residual_color = residual_to_colormap(residual)
            residual_resized = cv2.resize(residual_color, (w_orig, h_orig))
            
            # 统计
            valid_mask = mask & (gt_depth > MIN_DEPTH) & (gt_depth < MAX_DEPTH)
            abs_rel = np.mean(np.abs(residual[valid_mask]))
            
            # 标签
            model_label = f"{name.replace('sampling-', '').upper()} (rel={abs_rel:.3f})"
            
            cols.append(add_label(resize_keep_aspect(depth_color, target_h), model_label))
            cols.append(add_label(resize_keep_aspect(residual_resized, target_h), 'Error'))
        
        # 拼接
        combined = np.hstack(cols)
        cv2.imwrite(str(output_dir / f"{prefix}_comparison.png"), combined)
        print(f"Saved: {prefix}_comparison.png")

if __name__ == '__main__':
    main()
