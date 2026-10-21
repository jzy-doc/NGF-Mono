#!/usr/bin/env python3
"""
验证置信度估计与光度误差的关系

假设：置信度高的区域 = 重投影光度误差小（符合光度不变性假设）
验证方法：
1. 加载真实相邻帧
2. 用预测的深度和位姿进行图像重投影
3. 计算重投影光度误差
4. 检查置信度是否与光度误差负相关
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import readlines
from datasets.kitti_dataset import KITTIRAWDataset
import networks
from scipy.stats import pearsonr, spearmanr


def reproject_with_pose(src_img, depth, T, fx, fy, cx, cy):
    """
    用深度和位姿重投影源图像到目标视角
    src_img: [H, W, 3] 归一化图像
    depth: [H, W] 深度图
    T: [6] 位姿 [tx, ty, tz, rx, ry, rz] (从src到tgt的变换)
    """
    h, w = src_img.shape[:2]
    
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    u_grid, v_grid = np.meshgrid(u, v)
    
    x_norm = (u_grid - cx) / fx
    y_norm = (v_grid - cy) / fy
    
    X_c = x_norm * depth
    Y_c = y_norm * depth
    Z_c = depth
    
    # Rodrigues 公式
    theta = np.sqrt(T[3]**2 + T[4]**2 + T[5]**2 + 1e-8)
    rx, ry, rz = T[3]/theta, T[4]/theta, T[5]/theta
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    
    R = np.zeros((3, 3))
    R[0, 0] = cos_t + rx*rx*(1-cos_t)
    R[0, 1] = rx*ry*(1-cos_t) - rz*sin_t
    R[0, 2] = ry*sin_t + rx*rz*(1-cos_t)
    R[1, 0] = rz*sin_t + rx*ry*(1-cos_t)
    R[1, 1] = cos_t + ry*ry*(1-cos_t)
    R[1, 2] = -rx*sin_t + ry*rz*(1-cos_t)
    R[2, 0] = -ry*sin_t + rx*rz*(1-cos_t)
    R[2, 1] = rx*sin_t + ry*rz*(1-cos_t)
    R[2, 2] = cos_t + rz*rz*(1-cos_t)
    
    t = np.array([T[0], T[1], T[2]])
    
    points = np.stack([X_c.flatten(), Y_c.flatten(), Z_c.flatten()])
    world = R @ points + t.reshape(3, 1)
    
    X_w, Y_w, Z_w = world[0], world[1], world[2]
    valid = Z_w > 0.1
    
    u_proj = fx * X_w / (Z_w + 1e-8) + cx
    v_proj = fy * Y_w / (Z_w + 1e-8) + cy
    
    u_proj = u_proj.reshape(h, w)
    v_proj = v_proj.reshape(h, w)
    
    # 双线性插值
    warped = np.zeros_like(src_img)
    for y in range(h):
        for x in range(w):
            if valid[y*w+x]:
                u, v = u_proj[y, x], v_proj[y, x]
                u0, v0 = int(u), int(v)
                if 0 <= u0 < w-1 and 0 <= v0 < h-1:
                    du, dv = u - u0, v - v0
                    for c in range(3):
                        warped[y, x, c] = (1-du)*(1-dv)*src_img[v0, u0, c] + \
                                         du*(1-dv)*src_img[v0, u0+1, c] + \
                                         (1-du)*dv*src_img[v0+1, u0, c] + \
                                         du*dv*src_img[v0+1, u0+1, c]
    
    return warped


def main():
    data_path = '/data3/jzy/vision/monodepth2/kitti_data'
    load_weights_folder = '/data3/jzy/vision/my-mono/tmp/confmask/models/weights_49'
    
    encoder_path = os.path.join(load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(load_weights_folder, "depth.pth")
    confidence_path = os.path.join(load_weights_folder, "confidence.pth")
    pose_encoder_path = os.path.join(load_weights_folder, "pose_encoder.pth")
    pose_path = os.path.join(load_weights_folder, "pose.pth")
    
    encoder_dict = torch.load(encoder_path)
    decoder_dict = torch.load(decoder_path)
    confidence_dict = torch.load(confidence_path)
    pose_encoder_dict = torch.load(pose_encoder_path)
    pose_dict = torch.load(pose_path)
    
    img_ext = '.png'
    splits_dir = os.path.join(os.path.dirname(__file__), "splits")
    filenames = readlines(os.path.join(splits_dir, "eigen", "test_files_with_prev.txt"))
    
    h, w = encoder_dict['height'], encoder_dict['width']
    
    # 加载真实相邻帧: frame_ids=[-1, 0]
    dataset = KITTIRAWDataset(
        data_path, filenames, h, w,
        [-1, 0], 4, is_train=False, img_ext=img_ext)
    dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=2, pin_memory=True)
    
    encoder = networks.LiteMono(model='lite-mono', height=h, width=w, in_chans=3)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
    confidence_decoder = networks.DepthDecoder(
        encoder.num_ch_enc, scales=range(3), num_output_channels=2)
    pose_encoder = networks.ResnetEncoder(18, pretrained=False, num_input_images=2)
    pose_decoder = networks.PoseDecoder(
        pose_encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
    
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
    depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_decoder.state_dict()})
    confidence_decoder.load_state_dict({k: v for k, v in confidence_dict.items() 
                                       if k in confidence_decoder.state_dict()})
    pose_encoder.load_state_dict({k: v for k, v in pose_encoder_dict.items() if k in pose_encoder.state_dict()})
    pose_decoder.load_state_dict({k: v for k, v in pose_dict.items() if k in pose_decoder.state_dict()})
    
    encoder.cuda().eval()
    depth_decoder.cuda().eval()
    confidence_decoder.cuda().eval()
    pose_encoder.cuda().eval()
    pose_decoder.cuda().eval()
    
    save_dir = 'confidence_analysis'
    os.makedirs(save_dir, exist_ok=True)
    
    results_list = []
    all_correlations = []
    skip_count = 0
    
    print("-> Analyzing confidence vs photometric error relationship (real adjacent frames)")
    print("=" * 60)
    
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if idx >= 100:
                break
            
            # 当前帧和前一帧
            curr_rgb = data[("color", 0, 0)].cuda()
            prev_rgb = data[("color", -1, 0)].cuda()
            
            # 深度预测
            features = encoder(curr_rgb)
            depth_output = depth_decoder(features)
            pred_disp = depth_output[("disp", 0)]  # [1,1,H,W]
            
            # 置信度预测
            conf_output = confidence_decoder(features)
            pred_conf = conf_output[("disp", 0)]
            pred_conf = F.interpolate(pred_conf, size=(h, w), mode='bilinear', align_corners=False)
            pred_conf = torch.clamp(pred_conf[:, 0:1], min=1e-3, max=1.0)
            
            # 位姿预测 (curr -> prev)
            all_images = torch.cat([curr_rgb, prev_rgb], dim=1)
            all_features = [pose_encoder(all_images)]
            axisangle, translation = pose_decoder(all_features)
            
            # pose_decoder 输出: axisangle [1, 2, 1, 3], translation [1, 2, 1, 3]
            # 第一个是 prev 帧的位姿
            # T = [translation, axisangle] 形状 [6]
            T = torch.cat([translation[0, 0].flatten(), axisangle[0, 0].flatten()], dim=0).cpu().numpy()
            
            # 深度
            depth = (1.0 / (pred_disp[0, 0].cpu().numpy() + 1e-6))
            
            # numpy
            curr_rgb_np = curr_rgb[0].cpu().numpy().transpose(1, 2, 0)
            prev_rgb_np = prev_rgb[0].cpu().numpy().transpose(1, 2, 0)
            pred_conf_np = pred_conf[0, 0].cpu().numpy()
            
            # 相机内参（归一化到模型尺寸）
            fx = 0.58 * w
            fy = 1.92 * h
            cx = 0.5 * w
            cy = 0.5 * h
            
            # 重投影: 用深度和位姿将 prev warp 到 curr 视角
            # pose decoder 预测的是 curr 到 prev 的变换，需要取逆
            T_rev = -T  # 从 prev 到 curr 的近似逆变换
            
            warped = reproject_with_pose(prev_rgb_np, depth, T_rev, fx, fy, cx, cy)
            
            # 光度误差: curr vs warped(prev)
            photo_error = np.abs(curr_rgb_np - warped).mean(axis=2)
            
            # 有效像素
            valid_mask = (pred_conf_np > 0.01) & (photo_error > 0) & (photo_error < 2)
            
            if valid_mask.sum() > 500:
                errors = photo_error[valid_mask].flatten()
                confs = pred_conf_np[valid_mask].flatten()
                
                pearson_corr, _ = pearsonr(confs, errors)
                spearman_corr, _ = spearmanr(confs, errors)
                
                conf_bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
                bin_stats = []
                for low, high in conf_bins:
                    mask = (confs >= low) & (confs < high)
                    if mask.sum() > 50:
                        bin_stats.append({
                            'conf_range': f"{low:.1f}-{high:.1f}",
                            'mean_conf': confs[mask].mean(),
                            'mean_error': errors[mask].mean(),
                            'std_error': errors[mask].std(),
                            'count': mask.sum()
                        })
                
                result = {
                    'image_idx': idx,
                    'filename': filenames[idx],
                    'pearson_corr': pearson_corr,
                    'spearman_corr': spearman_corr,
                    'bin_stats': bin_stats,
                    'conf_map': pred_conf_np,
                    'photo_error': photo_error,
                    'rgb': curr_rgb_np,
                    'warped': warped
                }
                results_list.append(result)
                all_correlations.append(spearman_corr)
                
                if idx % 10 == 0:
                    print(f"Image {idx}: Spearman r = {spearman_corr:.4f}, Pearson r = {pearson_corr:.4f}")
    
    if skip_count > 0:
        print(f"\nSkipped {skip_count} images without previous frame")
    
    # 统计结果
    print("\n" + "=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)
    print(f"Total images analyzed: {len(all_correlations)}")
    print(f"Mean Spearman r: {np.mean(all_correlations):.4f}")
    print(f"Std Spearman r:  {np.std(all_correlations):.4f}")
    print(f"Mean Pearson r:  {np.mean([r['pearson_corr'] for r in results_list]):.4f}")
    
    neg_ratio = np.mean([c < 0 for c in all_correlations])
    print(f"Negative correlation ratio: {neg_ratio:.2%}")
    
    # 分组统计
    print("\n" + "-" * 60)
    print("CONFIDENCE BINS vs PHOTOMETRIC ERROR")
    print("-" * 60)
    print(f"{'Conf Range':>12} | {'Mean Conf':>10} | {'Mean Error':>10} | {'Std Error':>10}")
    print("-" * 60)
    
    all_bin_stats = {}
    for r in results_list:
        for bs in r['bin_stats']:
            key = bs['conf_range']
            if key not in all_bin_stats:
                all_bin_stats[key] = []
            all_bin_stats[key].append(bs['mean_error'])
    
    for conf_range in ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']:
        if conf_range in all_bin_stats:
            mean_err = np.mean(all_bin_stats[conf_range])
            std_err = np.std(all_bin_stats[conf_range])
            mid = (float(conf_range.split('-')[0]) + float(conf_range.split('-')[1])) / 2
            print(f"{conf_range:>12} | {mid:>10.2f} | {mean_err:>10.4f} | {std_err:>10.4f}")
    
    # 最好/最差可视化
    print("\n" + "=" * 60)
    print("BEST EXAMPLES (most negative correlation = assumption verified)")
    print("=" * 60)
    
    for rank, r in enumerate(sorted(results_list, key=lambda x: x['spearman_corr'])[:5]):
        print(f"\n#{rank+1}: {r['filename']}")
        print(f"    Spearman: {r['spearman_corr']:.4f}, Pearson: {r['pearson_corr']:.4f}")
        for bs in r['bin_stats']:
            print(f"    {bs['conf_range']}: error={bs['mean_error']:.4f} (n={bs['count']})")
        
        rgb = r['rgb']
        conf = r['conf_map']
        photo_err = r['photo_error']
        warped = r['warped']
        
        rgb_vis = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(f"{save_dir}/best_{rank+1:02d}_rgb.png", cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR))
        
        conf_vis = cv2.applyColorMap((np.clip(conf, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(f"{save_dir}/best_{rank+1:02d}_conf.png", conf_vis)
        
        err_vis = np.clip(photo_err * 5, 0, 1) * 255
        err_color = cv2.applyColorMap(err_vis.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(f"{save_dir}/best_{rank+1:02d}_photo_error.png", err_color)
        
        warped_vis = np.clip(warped * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(f"{save_dir}/best_{rank+1:02d}_warped.png", cv2.cvtColor(warped_vis, cv2.COLOR_RGB2BGR))
        
        overlay = cv2.addWeighted(cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR), 0.5, conf_vis, 0.5, 0)
        cv2.imwrite(f"{save_dir}/best_{rank+1:02d}_overlay.png", overlay)
        
        print(f"    -> Saved to {save_dir}/best_{rank+1:02d}_*.png")
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    
    mean_corr = np.mean(all_correlations)
    if mean_corr < -0.15:
        print(f"✓ 置信度与光度误差呈显著负相关 (r={mean_corr:.4f})")
        print("  假设得到验证：置信度高的区域光度误差小")
    elif mean_corr < -0.05:
        print(f"○ 置信度与光度误差呈负相关 (r={mean_corr:.4f})")
        print("  假设部分验证")
    elif mean_corr < 0:
        print(f"△ 置信度与光度误差呈弱负相关 (r={mean_corr:.4f})")
        print("  假设验证效果较弱")
    else:
        print(f"✗ 置信度与光度误差呈正相关或无关 (r={mean_corr:.4f})")
        print("  假设未验证")
    
    print("\n-> Done!")


if __name__ == "__main__":
    main()
