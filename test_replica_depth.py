"""
测试 my-mono 预训练模型在 Replica 数据集上的深度估计质量。
- 分辨率适配测试：RGB 1200×680 → resize → 模型输入 192×640 → 输出 disparity → resize 回 1200×680
- 深度质量评估：与 Replica 真实深度对比，允许 scale 差异（中位数对齐）

用法:
    python test_replica_depth.py
"""

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import time
import json

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
import PIL.Image as pil

# 加载 my-mono 环境
sys.path.insert(0, '/data3/jzy/vision/my-mono')
import networks

# =============================================================================
# 配置
# =============================================================================

REPLICA_ROOT = '/data3/jzy/slam/nice-slam/Datasets/Replica'
MODEL_ROOT = '/data3/jzy/vision/my-mono/tmp/edge_midfusion_gate/models/weights_29'
# MODEL_ROOT = '/data3/jzy/vision/my-mono/tmp/edge_film/models/weights_1'  # 可切换对比

# Replica 相机参数 (来自 configs/Replica/replica.yaml)
PNG_DEPTH_SCALE = 6553.5   # depth_meters = depth_uint16 / 6553.5
DEPTH_MIN_METERS = 0.1
DEPTH_MAX_METERS = 100.0

# 测试配置
TEST_SCENES = ['office0', 'office1', 'room0']          # 测试场景
TEST_FRAMES = [0, 200, 500, 1000, 1500, 1999]         # 每个场景采样的帧索引
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# 深度评估指标（允许 scale 差异的版本）
# =============================================================================

def disp_to_depth(disp, min_depth, max_depth):
    """将 disparity 转为 depth"""
    min_disp = 1.0 / max_depth
    max_disp = 1.0 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1.0 / (scaled_disp + 1e-8)
    return depth


def align_scale(pred, gt):
    """
    通过中位数对齐消除 global scale 差异。
    pred_scaled = pred * (median(gt) / median(pred))
    """
    median_pred = np.median(pred[pred > 0])
    median_gt = np.median(gt[gt > 0])
    if median_pred < 1e-6:
        return pred
    return pred * (median_gt / median_pred)


def compute_depth_errors(pred, gt, min_depth=0.1, max_depth=10.0):
    """
    计算深度估计误差指标。
    pred 和 gt 形状相同，单位都是米。
    """
    # 只考虑有效区域（gt > 0, 且在合理范围内）
    valid_mask = (gt > min_depth) & (gt < max_depth) & (pred > 0)
    if valid_mask.sum() < 10:
        return None

    pred = pred[valid_mask]
    gt = gt[valid_mask]

    # Scale-aligned 版本（允许 global scale 差异）
    pred_aligned = align_scale(pred, gt)

    thresh = np.maximum((pred_aligned / gt), (gt / pred_aligned))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.abs(pred_aligned - gt) / gt
    sq_rel = ((pred_aligned - gt) ** 2) / gt

    rmse = np.sqrt(((pred_aligned - gt) ** 2).mean())
    rmse_log = np.sqrt(((np.log(pred_aligned) - np.log(gt)) ** 2).mean())

    abs_rel_mean = abs_rel.mean()
    sq_rel_mean = sq_rel.mean()
    rmse_mean = rmse

    return {
        'abs_rel': abs_rel_mean,
        'sq_rel': sq_rel_mean,
        'rmse': rmse_mean,
        'rmse_log': rmse_log,
        'a1': a1,
        'a2': a2,
        'a3': a3,
        'valid_pixels': valid_mask.sum(),
    }


# =============================================================================
# 模型加载
# =============================================================================

def load_depth_model(model_path):
    """加载 my-mono 预训练深度模型"""
    encoder_path = os.path.join(model_path, 'encoder.pth')
    decoder_path = os.path.join(model_path, 'depth.pth')

    assert os.path.exists(encoder_path), f"encoder.pth not found at {encoder_path}"
    assert os.path.exists(decoder_path), f"depth.pth not found at {decoder_path}"

    encoder_dict = torch.load(encoder_path, map_location='cpu')
    decoder_dict = torch.load(decoder_path, map_location='cpu')

    feed_height = encoder_dict['height']
    feed_width = encoder_dict['width']

    print(f"  模型训练分辨率: {feed_height} × {feed_width}")

    encoder = networks.LiteMono(
        model=encoder_dict.get('model', 'lite-mono'),
        height=feed_height,
        width=feed_width,
        drop_path=0.0  # inference 时关闭 dropout
    )

    encoder.load_state_dict({k: v for k, v in encoder_dict.items()
                             if k in encoder_dict and k in encoder.state_dict()},
                            strict=False)
    encoder.to(DEVICE)
    encoder.eval()

    decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
    decoder.load_state_dict({k: v for k, v in decoder_dict.items()
                             if k in decoder_dict and k in decoder.state_dict()},
                            strict=False)
    decoder.to(DEVICE)
    decoder.eval()

    return encoder, decoder, feed_height, feed_width


# =============================================================================
# 单帧深度预测
# =============================================================================

def predict_depth_from_pil(img_pil, encoder, decoder, feed_height, feed_width):
    """
    预测单帧深度（输入已经是 PIL Image）。

    Args:
        img_pil: PIL Image
        encoder, decoder: 深度模型
        feed_height, feed_width: 模型训练分辨率

    Returns:
        depth: numpy array (orig_H, orig_W), 单位米
    """
    # 1. resize 到模型输入分辨率
    orig_w, orig_h = img_pil.size  # PIL: (width, height)
    img_resized = img_pil.resize((feed_width, feed_height), pil.LANCZOS)
    img_tensor = transforms.ToTensor()(img_resized).unsqueeze(0).to(DEVICE)

    # 2. 预测 disparity
    with torch.no_grad():
        features = encoder(img_tensor)
        outputs = decoder(features)
        disp = outputs[("disp", 0)]  # (1, 1, feed_H, feed_W)

    # 3. disparity → depth
    depth_pred = disp_to_depth(disp, DEPTH_MIN_METERS, DEPTH_MAX_METERS)

    # 4. resize 回原始分辨率
    depth_resized = F.interpolate(
        depth_pred,
        size=(orig_h, orig_w),  # (H, W)
        mode='bilinear',
        align_corners=False
    ).squeeze().cpu().numpy()

    return depth_resized


# =============================================================================
# 主测试流程
# =============================================================================

def main():
    print("=" * 70)
    print("Replica 深度估计质量测试")
    print("=" * 70)
    print(f"设备: {DEVICE}")
    print(f"模型: {MODEL_ROOT}")
    print()

    # -------------------------------------------------------------------------
    # 1. 加载模型
    # -------------------------------------------------------------------------
    print("[1/4] 加载深度模型...")
    encoder, decoder, feed_h, feed_w = load_depth_model(MODEL_ROOT)
    print(f"  模型输入分辨率: {feed_h} × {feed_w}")
    print()

    # -------------------------------------------------------------------------
    # 2. 收集测试帧
    # -------------------------------------------------------------------------
    print("[2/4] 收集测试帧...")
    test_frames = []
    for scene in TEST_SCENES:
        scene_root = os.path.join(REPLICA_ROOT, scene, 'results')
        if not os.path.exists(scene_root):
            print(f"  警告: {scene_root} 不存在，跳过")
            continue

        color_files = sorted(glob.glob(os.path.join(scene_root, 'frame*.jpg')))
        depth_files = sorted(glob.glob(os.path.join(scene_root, 'depth*.png')))

        if not color_files:
            print(f"  警告: {scene} 没有找到帧，跳过")
            continue

        n_frames = len(color_files)
        print(f"  场景 {scene}: {n_frames} 帧, 分辨率 {1200}×{680}")

        for frame_idx in TEST_FRAMES:
            if frame_idx >= n_frames:
                continue
            color_path = color_files[frame_idx]
            depth_path = depth_files[frame_idx]
            frame_num = int(os.path.basename(color_path).replace('frame', '').replace('.jpg', ''))
            test_frames.append({
                'scene': scene,
                'frame_idx': frame_idx,
                'frame_num': frame_num,
                'color_path': color_path,
                'depth_path': depth_path,
            })

    print(f"  共收集 {len(test_frames)} 个测试帧")
    print()

    # -------------------------------------------------------------------------
    # 3. 预测 + 评估
    # -------------------------------------------------------------------------
    print("[3/4] 逐帧预测深度并评估...")
    all_results = []
    total_time = 0.0

    for i, item in enumerate(test_frames):
        scene = item['scene']
        frame_num = item['frame_num']
        color_path = item['color_path']
        depth_path = item['depth_path']

        # 读取 ground truth depth
        gt_depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        gt_depth = gt_depth_raw / PNG_DEPTH_SCALE  # 转为米

        # 预测深度
        t0 = time.time()
        img_pil = pil.open(color_path).convert('RGB')
        pred_depth = predict_depth_from_pil(img_pil, encoder, decoder, feed_h, feed_w)
        t1 = time.time()
        inference_time = (t1 - t0) * 1000  # ms

        # resize gt depth 到模型输出分辨率（pred 经过了 resize，所以要对齐）
        # pred 被 resize 回了 1200×680，gt 也是 1200×680，无需 resize

        # 计算误差
        errors = compute_depth_errors(pred_depth, gt_depth)
        if errors is None:
            print(f"  [{i+1}/{len(test_frames)}] {scene}/frame{frame_num:06d}: 有效像素不足，跳过")
            continue

        errors['scene'] = scene
        errors['frame_num'] = frame_num
        errors['inference_ms'] = inference_time
        errors['pred_median'] = float(np.median(pred_depth[pred_depth > 0]))
        errors['gt_median'] = float(np.median(gt_depth[gt_depth > 0]))
        errors['scale_factor'] = errors['gt_median'] / errors['pred_median']
        all_results.append(errors)
        total_time += inference_time

        print(f"  [{i+1}/{len(test_frames)}] {scene}/frame{frame_num:06d} | "
              f"AbsRel={errors['abs_rel']:.4f} | "
              f"RMSE={errors['rmse']:.3f}m | "
              f"A1@1.25={errors['a1']:.3f} | "
              f"scale={errors['scale_factor']:.3f} | "
              f"t={inference_time:.1f}ms")

    print()

    # -------------------------------------------------------------------------
    # 4. 汇总统计
    # -------------------------------------------------------------------------
    print("[4/4] 汇总统计...")
    print("=" * 70)
    print(f"{'场景':<12} {'帧':<8} {'AbsRel':<10} {'SqRel':<10} {'RMSE':<10} "
          f"{'RMSE_log':<10} {'A1':<8} {'A2':<8} {'A3':<8}")
    print("-" * 70)

    scene_stats = {}
    for r in all_results:
        scene = r['scene']
        if scene not in scene_stats:
            scene_stats[scene] = []
        scene_stats[scene].append(r)

    for scene, results in sorted(scene_stats.items()):
        n = len(results)
        avg = {k: np.mean([r[k] for r in results]) for k in ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']}
        print(f"{scene:<12} n={n:<6} {avg['abs_rel']:.4f}     {avg['sq_rel']:.4f}     "
              f"{avg['rmse']:.3f}    {avg['rmse_log']:.4f}    {avg['a1']:.3f}    {avg['a2']:.3f}    {avg['a3']:.3f}")

    print("-" * 70)
    n_total = len(all_results)
    if n_total > 0:
        overall = {k: np.mean([r[k] for r in all_results]) for k in
                   ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3', 'inference_ms', 'scale_factor']}
        print(f"{'OVERALL':<12} n={n_total:<6} {overall['abs_rel']:.4f}     {overall['sq_rel']:.4f}     "
              f"{overall['rmse']:.3f}    {overall['rmse_log']:.4f}    {overall['a1']:.3f}    "
              f"{overall['a2']:.3f}    {overall['a3']:.3f}")
        print(f"\n  平均推理时间: {overall['inference_ms']:.1f} ms/帧")
        print(f"  总推理时间: {total_time/1000:.2f} 秒")

    print()
    print("=" * 70)
    print("分析结论:")
    print("=" * 70)

    if n_total > 0:
        abs_rel = overall['abs_rel']
        rmse = overall['rmse']
        a1 = overall['a1']

        # 分辨率适配评估
        print(f"\n【分辨率适配】")
        print(f"  Replica 图像分辨率: 1200 × 680 (宽高比 H:W ≈ 0.57)")
        print(f"  模型训练分辨率:     {feed_h} × {feed_w} (宽高比 H:W ≈ {feed_h/feed_w:.2f})")
        print(f"  宽高比差异: {abs(0.57 - feed_h/feed_w):.2f} (宽高比失真程度)")
        print(f"  resize 方向: 1200×680 → {feed_h}×{feed_w} → 1200×680 (双线性插值)")

        if abs_rel < 0.15:
            quality = "优秀"
        elif abs_rel < 0.25:
            quality = "良好"
        elif abs_rel < 0.40:
            quality = "一般"
        else:
            quality = "较差"

        print(f"\n【深度质量评估】")
        print(f"  AbsRel = {abs_rel:.4f} ({quality})")
        print(f"  RMSE   = {rmse:.3f}m")
        print(f"  A1     = {a1:.3f} (δ < 1.25 的比例)")
        overall_scale = np.mean([r['scale_factor'] for r in all_results])
        print(f"  Scale 中位数比 = {overall_scale:.3f}x "
              f"(预测深度 scale vs 真实深度)")
        print(f"\n  说明: AbsRel < 0.15 表示优秀, < 0.25 良好, < 0.40 一般")
        print(f"  注意: 单目深度在室内场景通常 AbsRel 在 0.15~0.35 之间，")
        print(f"        与 RGB-D 传感器直接测量的精度(通常 < 0.05) 有本质差距")

        # NICE-SLAM 兼容性评估
        print(f"\n【NICE-SLAM 兼容性评估】")
        # NICE-SLAM 主要用深度做 sampling guidance 和 depth loss
        # AbsRel = 0.2 意味着平均 20% 的深度误差
        # 对于 surface sampling (near=depth*0.01, far=depth*1.2):
        #   - near=2m 时, 误差 0.4m → 采样范围 [0.02, 2.4], 影响可控
        #   - near=5m 时, 误差 1.0m → 采样范围 [0.05, 6.0], 误差放大
        if abs_rel < 0.20:
            compat = "兼容（深度质量足够用于 sampling guidance）"
        elif abs_rel < 0.35:
            compat = "勉强兼容（需增加表面采样点数量缓解误差）"
        else:
            compat = "不兼容（建议用真实深度或改善单目深度精度）"
        print(f"  评估结果: {compat}")
        print(f"  建议: 结合 NICE-SLAM 的 uncertainty 机制过滤不可靠深度区域")

    print()
    print("=" * 70)


if __name__ == '__main__':
    main()
