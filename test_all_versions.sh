#!/bin/bash
# 测试所有 edge_midfusion 版本

conda activate slam
cd /data3/jzy/vision/my-mono

echo "=========================================="
echo "1. Lite-Mono Baseline (不带边信息)"
echo "=========================================="
python evaluate_depth.py \
  --data_path /data3/jzy/vision/monodepth2/kitti_data \
  --load_weights_folder /data3/jzy/vision/Lite-Mono/tmp/mytrain/models/weights_29 \
  --png

echo ""
echo "=========================================="
echo "2. edge_midfusion (原始gate版本)"
echo "=========================================="
python evaluate_depth.py \
  --data_path /data3/jzy/vision/monodepth2/kitti_data \
  --load_weights_folder /data3/jzy/vision/my-mono/tmp/edge_midfusion/models/weights_29 \
  --use_edge_mid_fusion \
  --png

echo ""
echo "=========================================="
echo "3. edge_midfusion_v2"
echo "=========================================="
python evaluate_depth.py \
  --data_path /data3/jzy/vision/monodepth2/kitti_data \
  --load_weights_folder /data3/jzy/vision/my-mono/tmp/edge_midfusion_v2/models/weights_29 \
  --use_edge_mid_fusion \
  --png

echo ""
echo "=========================================="
echo "4. edge_midfusion_v3"
echo "=========================================="
python evaluate_depth.py \
  --data_path /data3/jzy/vision/monodepth2/kitti_data \
  --load_weights_folder /data3/jzy/vision/my-mono/tmp/edge_midfusion_v3/models/weights_29 \
  --use_edge_mid_fusion \
  --png

echo ""
echo "=========================================="
echo "5. edge_midfusion_v4"
echo "=========================================="
python evaluate_depth.py \
  --data_path /data3/jzy/vision/monodepth2/kitti_data \
  --load_weights_folder /data3/jzy/vision/my-mono/tmp/edge_midfusion_v4/models/weights_29 \
  --use_edge_mid_fusion \
  --png

echo ""
echo "=========================================="
echo "全部测试完成!"
echo "=========================================="
