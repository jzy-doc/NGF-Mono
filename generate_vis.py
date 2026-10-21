"""
用光流计算光度误差
"""
import os
import numpy as np
import cv2
from pathlib import Path

def compute_photometric_error_with_flow(img1, img2, flow):
    """用光流计算重投影光度误差"""
    h, w = img1.shape[:2]
    
    flow_u = flow[:, :, 0]
    flow_v = flow[:, :, 1]
    
    u_target = np.arange(w)[None, :] + flow_u
    v_target = np.arange(h)[:, None] + flow_v
    
    valid = (u_target >= 0) & (u_target < w) & (v_target >= 0) & (v_target < h)
    
    warped = np.zeros_like(img1, dtype=np.float32)
    
    u_f = u_target.astype(np.float32)
    v_f = v_target.astype(np.float32)
    
    for y in range(h):
        for x in range(w):
            if valid[y, x]:
                u, v = u_f[y, x], v_f[y, x]
                u0, v0 = int(u), int(v)
                u1, v1 = min(u0 + 1, w - 1), min(v0 + 1, h - 1)
                du, dv = u - u0, v - v0
                
                if img2.ndim == 3:
                    for c in range(3):
                        warped[y, x, c] = (1-du)*(1-dv)*img2[v0, u0, c] + \
                                          du*(1-dv)*img2[v0, u1, c] + \
                                          (1-du)*dv*img2[v1, u0, c] + \
                                          du*dv*img2[v1, u1, c]
                else:
                    warped[y, x] = (1-du)*(1-dv)*img2[v0, u0] + \
                                   du*(1-dv)*img2[v0, u1] + \
                                   (1-du)*dv*img2[v1, u0] + \
                                   du*dv*img2[v1, u1]
    
    # 重投影误差 = warp后的图与目标图的差异
    error = np.abs(warped - img2.astype(np.float32))
    
    return error, warped, valid

def estimate_flow(img1, img2):
    """用 Farneback 光流估计"""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0
    )
    return flow

def main():
    base_dir = Path('/data3/jzy/vision/monodepth2/kitti_data')
    output_dir = Path('/data3/jzy/vision/my-mono/vis_results')
    output_dir.mkdir(exist_ok=True)
    
    scenarios = [
        {'name': 'Example 1: Occlusion & Depth Discontinuity', 
         'seq': '2011_09_26/2011_09_26_drive_0011_sync', 'frame': 14, 'frame_next': 15},
        {'name': 'Example 2: Illumination Variation', 
         'seq': '2011_09_26/2011_09_26_drive_0009_sync', 'frame': 124, 'frame_next': 125},
        {'name': 'Example 3: Non-Lambertian Surface', 
         'seq': '2011_09_30/2011_09_30_drive_0016_sync', 'frame': 200, 'frame_next': 201},
    ]
    
    for i, scenario in enumerate(scenarios):
        seq_path = base_dir / scenario['seq']
        img_dir = seq_path / 'image_02' / 'data'
        
        frame_idx = scenario['frame']
        frame_next = scenario['frame_next']
        
        img1_path = img_dir / f'{frame_idx:010d}.png'
        img2_path = img_dir / f'{frame_next:010d}.png'
        
        if not img1_path.exists() or not img2_path.exists():
            print(f"Frames not found")
            continue
        
        img_ref = cv2.imread(str(img1_path))
        img_target = cv2.imread(str(img2_path))
        
        if img_ref is None or img_target is None:
            continue
        
        h, w = img_ref.shape[:2]
        
        # 估计光流
        flow = estimate_flow(img_ref, img_target)
        
        # 转灰度
        img_ref_gray = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
        img_target_gray = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # 计算重投影光度误差
        reproj_error, warped, valid = compute_photometric_error_with_flow(
            img_ref_gray, img_target_gray, flow
        )
        
        # 缩放
        scale = 3
        new_w = w // scale
        new_h = h // scale
        
        img_ref_small = cv2.resize(img_ref, (new_w, new_h))
        img_target_small = cv2.resize(img_target, (new_w, new_h))
        
        # 光度误差图
        reproj_display = np.clip(reproj_error / 40.0, 0, 1) * 255
        reproj_color = cv2.applyColorMap(reproj_display.astype(np.uint8), cv2.COLORMAP_JET)
        reproj_color = cv2.resize(reproj_color, (new_w, new_h))
        
        # 拼接: 参考 | 目标 | 光度误差
        result = np.hstack([img_ref_small, img_target_small, reproj_color])
        
        # 标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        color = (255, 255, 255)
        
        cv2.putText(result, 'Reference (t)', (10, 18), font, font_scale, color, thickness)
        cv2.putText(result, 'Target (t+1)', (new_w + 10, 18), font, font_scale, color, thickness)
        cv2.putText(result, 'Photometric Error', (2*new_w + 10, 18), font, font_scale, color, thickness)
        
        # 标题
        cv2.putText(result, scenario['name'], (10, new_h - 8), 
                    font, font_scale * 0.9, (0, 255, 0), thickness)
        
        output_path = output_dir / f'photo_inconsistency_{i+1}.png'
        cv2.imwrite(str(output_path), result)
        print(f"Saved: {output_path}")

if __name__ == '__main__':
    main()
