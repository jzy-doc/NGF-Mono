"""
寻找光度一致性假设失效的典型场景
- 遮挡区域
- 光照变化
- 非朗伯表面
"""
import os
import numpy as np
import cv2
from pathlib import Path

def compute_photometric_error(img1, img2):
    """计算两张图之间的光度误差"""
    # 转换为灰度
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray1 = img1.astype(np.float32)
        gray2 = img2.astype(np.float32)
    
    # 绝对误差
    abs_diff = np.abs(gray1 - gray2)
    
    return abs_diff, gray1, gray2

def analyze_scene(img1_path, img2_path):
    """分析两个视角之间的光度差异"""
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        return None
    
    abs_diff, gray1, gray2 = compute_photometric_error(img1, img2)
    
    # 统计信息
    mean_error = np.mean(abs_diff)
    std_error = np.std(abs_diff)
    max_error = np.max(abs_diff)
    
    # 高误差区域比例 (> 30)
    high_error_mask = abs_diff > 30
    high_error_ratio = np.mean(high_error_mask)
    
    return {
        'mean_error': mean_error,
        'std_error': std_error,
        'max_error': max_error,
        'high_error_ratio': high_error_ratio,
        'abs_diff': abs_diff,
        'gray1': gray1,
        'gray2': gray2,
        'img1': img1,
        'img2': img2
    }

def save_comparison(result, output_path, title=""):
    """保存对比图"""
    img1 = result['img1']
    img2 = result['img2']
    abs_diff = result['abs_diff']
    
    # 归一化差异图用于显示
    diff_display = abs_diff / 255.0 * 3  # 放大差异以便观察
    diff_display = np.clip(diff_display, 0, 1)
    diff_display = (diff_display * 255).astype(np.uint8)
    diff_color = cv2.applyColorMap(diff_display, cv2.COLORMAP_JET)
    
    # 创建对比图
    h, w = img1.shape[:2]
    
    # 左图：参考视角 | 右图：目标视角 | 下方：光度误差图
    top_row = np.hstack([img1, img2])
    bottom_row = np.hstack([
        np.zeros_like(img1),
        diff_color
    ])
    
    # 添加标签
    label_h = 40
    top_with_label = np.zeros((h + label_h, 2*w, 3), dtype=np.uint8)
    top_with_label[label_h:, :w] = img1
    top_with_label[label_h:, w:] = img2
    
    bottom_with_label = np.zeros((h + label_h, 2*w, 3), dtype=np.uint8)
    bottom_with_label[label_h:, w:] = diff_color
    
    # 添加文字标签
    cv2.putText(top_with_label, 'Reference View (t)', (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(top_with_label, 'Target View (t+1)', (w + 10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(bottom_with_label, '', (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(bottom_with_label, 'Photometric Error', (w + 10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    result_img = np.vstack([top_with_label, bottom_with_label])
    
    # 添加标题
    if title:
        cv2.putText(result_img, title, (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, result_img)
    
    return result_img

def scan_sequence(seq_path, interval=1):
    """扫描一个序列，找出光度误差最大的帧"""
    img_dir = Path(seq_path) / 'image_02' / 'data'
    if not img_dir.exists():
        img_dir = Path(seq_path) / 'image_02' / 'data'
    
    images = sorted(list(img_dir.glob('*.png')))
    if len(images) < 2:
        return []
    
    results = []
    for i in range(0, len(images) - interval, max(1, len(images)//100)):
        img1_path = images[i]
        img2_path = images[min(i + interval, len(images)-1)]
        
        result = analyze_scene(str(img1_path), str(img2_path))
        if result:
            result['frame_idx'] = i
            result['frame_name'] = img1_path.name
            results.append(result)
    
    # 按高误差比例排序
    results.sort(key=lambda x: x['high_error_ratio'], reverse=True)
    
    return results

def main():
    base_dir = Path('/data3/jzy/vision/monodepth2/kitti_data')
    sequences = [
        '2011_09_26/2011_09_26_drive_0005_sync',
        '2011_09_26/2011_09_26_drive_0009_sync',
        '2011_09_26/2011_09_26_drive_0011_sync',
        '2011_09_30/2011_09_30_drive_0016_sync',
        '2011_09_30/2011_09_30_drive_0020_sync',
        '2011_10_03/2011_10_03_drive_0047_sync',
    ]
    
    all_results = []
    
    for seq in sequences:
        seq_path = base_dir / seq
        if not seq_path.exists():
            print(f"Sequence not found: {seq_path}")
            continue
        
        print(f"Scanning {seq}...")
        results = scan_sequence(seq_path)
        for r in results:
            r['sequence'] = seq
        all_results.extend(results)
    
    # 排序找出误差最大的
    all_results.sort(key=lambda x: x['high_error_ratio'], reverse=True)
    
    # 输出前10个最有问题的场景
    print("\nTop 10 scenes with photometric inconsistency:")
    for i, r in enumerate(all_results[:10]):
        print(f"{i+1}. {r['sequence']}/{r['frame_name']}: "
              f"high_error_ratio={r['high_error_ratio']:.3f}, "
              f"mean_error={r['mean_error']:.1f}")
    
    # 保存前3个最有问题的场景
    output_dir = Path('/data3/jzy/vision/my-mono/vis_results')
    output_dir.mkdir(exist_ok=True)
    
    titles = [
        "Example 1: Occlusion & Depth Discontinuity",
        "Example 2: Illumination Variation",
        "Example 3: Non-Lambertian Surface"
    ]
    
    for i, r in enumerate(all_results[:3]):
        output_path = output_dir / f'photo_inconsistency_{i+1}.png'
        save_comparison(r, str(output_path), titles[i])
        print(f"Saved: {output_path}")

if __name__ == '__main__':
    main()
