python train.py --data_path /home/desc/jzy/robot_vision/2.5D/monodepth2/kitti_data --model_name mytrain --num_epochs 30 --batch_size 12 --weights_init scratch --lr 0.0001 5e-6 31 0.0001 1e-5 31 --png

CUDA_VISIBLE_DEVICES=2 python train.py --model_name gaussian4ch --data_path /data3/jzy/vision/monodepth2/kitti_data --model_name edge_guass --png --use_edge_input --fourth_channel_mode gaussian --fourth_channel_noise_std 0.2 --num_epochs 30 --batch_size 12 --weights_init scratch --lr 0.0001 5e-6 31 0.0001 1e-5 31


# depth测试
python evaluate_depth.py --data_path /data3/jzy/vision/monodepth2/kitti_data --load_weights_folder /data3/jzy/vision/my-mono/tmp/edge_guass/models/weights_29 --png --use_edge_input 

python evaluate_depth.py --data_path /data3/jzy/vision/monodepth2/kitti_data --load_weights_folder /data3/jzy/vision/my-mono/tmp/edge_guass/models/weights_29 --png --use_edge_input 

python evaluate_depth.py --data_path /data3/jzy/vision/monodepth2/kitti_data --load_weights_folder /data3/jzy/vision/my-mono/tmp/confmask/models/weights_49 --png

python evaluate_depth.py --data_path /data3/jzy/vision/monodepth2/kitti_data --load_weights_folder /data3/jzy/vision/my-mono/tmp/edge_midfusion/models/weights_29 --png --use_edge_mid_fusion --edge_fusion_strength 1.0


以下命令在 my-mono 目录执行（把数据路径替换掉）：

基线（原始逻辑）
python train.py --data_path /data3/jzy/vision/monodepth2/kitti_data --model_name baseline
仅改进1：Sobel 第4通道
python train.py --data_path /data3/jzy/vision/monodepth2/kitti_data --model_name edge4ch --use_edge_input
仅改进2：confidence 替代 auto-mask
CUDA_VISIBLE_DEVICES=1 python train.py --data_path /data3/jzy/vision/monodepth2/kitti_data --model_name confmask --use_confidence_mask --confidence_reg_weight 0.01 --num_epochs 50 --batch_size 24 --weights_init scratch --lr 0.0001 5e-6 31 0.0001 1e-5 31 --png 
两个都开
python train.py --data_path /data3/jzy/vision/monodepth2/kitti_data --model_name edge4ch_conf --use_edge_input --use_confidence_mask
调 confidence 正则强度（建议试 0.005 / 0.01 / 0.02）
python train.py --data_path /data3/jzy/vision/monodepth2/kitti_data --model_name confmask_w001 --use_confidence_mask --confidence_reg_weight 0.01


1) 训练中层融合模型（推荐）
CUDA_VISIBLE_DEVICES=3 python train.py \
  --data_path /data3/jzy/vision/monodepth2/kitti_data \
  --model_name edge_midfusion \
  --use_edge_mid_fusion \
  --edge_fusion_strength 1.0 \
  --num_epochs 30 \
  --batch_size 12 \
  --weights_init scratch \
  --lr 0.0001 5e-6 31 0.0001 1e-5 31 \
  --png

改进版（门控）

CUDA_VISIBLE_DEVICES=3 python train.py \
  --data_path /data3/jzy/vision/monodepth2/kitti_data \
  --model_name edge_midfusion \
  --use_edge_mid_fusion \
  --num_epochs 30 \
  --batch_size 12 \
  --weights_init scratch \
  --lr 0.0001 5e-6 31 0.0001 1e-5 31 \
  --png

2) 评估中层融合模型

python evaluate_depth.py \
  --data_path /data3/jzy/vision/monodepth2/kitti_data \
  --load_weights_folder /home/desc/jzy/robot_vision/2.5D/my-mono/tmp/edge_midfusion/models/weights_29 \
  --use_edge_mid_fusion \
  --edge_fusion_strength 1.0 \
  --png

# 训练使用dino
CUDA_VISIBLE_DEVICES=2 python train.py \
  --data_path /data3/jzy/vision/monodepth2/kitti_data \
  --model_name dino_min_replace \
  --depth_encoder_type dino \
  --dino_pretrained \
  --num_epochs 30 \
  --batch_size 12 \
  --weights_init scratch \
  --lr 0.0001 5e-6 31 0.0001 1e-5 31 \
  --png --height 224 \
  --width 672 --dino_model_name dinov2_vits14

python evaluate_depth.py \
  --data_path /data3/jzy/vision/monodepth2/kitti_data \
  --load_weights_folder /data3/jzy/vision/my-mono/tmp/dino_min_replace/models/weights_29 \
  --depth_encoder_type dino \
  --png --dino_model_name dinov2_vits14


输入图像 → Stem → Stage 0 (1/4) → Stage 1 (1/8) → Stage 2 (1/16) → 特征输出
              DilatedConv×4     DilatedConv×4    LGFI + DilatedConv×9

RGB ─→ Stem ─→ Stage 0 ─→ Fuse_4 ─→ Stage 1 ─→ Fuse_8 ─→ Stage 2 ─→ Fuse_16 ─→ 融合特征
                ↑              ↑             ↑              ↑
Sobel ─→ edge_down_4 ─→ edge_down_8 ─→ edge_down_16    (独立边界分支)
            │              │             │
          gate_4         gate_8        gate_16         (门控网络)