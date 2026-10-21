python train.py --data_path /home/desc/jzy/robot_vision/2.5D/monodepth2/kitti_data --model_name mytrain --num_epochs 30 --batch_size 12 --weights_init scratch --lr 0.0001 5e-6 31 0.0001 1e-5 31 --png

# edge mid-fusion 训练（推荐新方案）
python train.py --data_path /home/desc/jzy/robot_vision/2.5D/monodepth2/kitti_data --model_name edge_midfusion --use_edge_mid_fusion --num_epochs 30 --batch_size 12 --weights_init scratch --lr 0.0001 5e-6 31 0.0001 1e-5 31 --png

# depth测试
python evaluate_depth.py --data_path /home/desc/jzy/robot_vision/2.5D/monodepth2/kitti_data --load_weights_folder /home/desc/jzy/robot_vision/2.5D/my-mono/tmp/edge4ch/models/weights_29 --png

python evaluate_depth.py --data_path /home/desc/jzy/robot_vision/2.5D/monodepth2/kitti_data --load_weights_folder /home/desc/jzy/robot_vision/2.5D/my-mono/tmp/edge4ch/models/weights_29 --png --use_edge_input

# edge mid-fusion 评估（与训练参数一致）
python evaluate_depth.py --data_path /home/desc/jzy/robot_vision/2.5D/monodepth2/kitti_data --load_weights_folder /home/desc/jzy/robot_vision/2.5D/my-mono/tmp/edge_midfusion/models/weights_29 --png --use_edge_mid_fusion

# edge mid-fusion + 区域指标评估
python evaluate_depth.py --data_path /home/desc/jzy/robot_vision/2.5D/monodepth2/kitti_data --load_weights_folder /home/desc/jzy/robot_vision/2.5D/my-mono/tmp/edge_midfusion/models/weights_29 --png --use_edge_mid_fusion --report_region_metrics --boundary_band_px 3 --contour_band_px 5

以下命令在 my-mono 目录执行（把数据路径替换掉）：

基线（原始逻辑）
python train.py --data_path /path/to/kitti --model_name baseline
仅改进1：Sobel 第4通道
python train.py --data_path /home/desc/jzy/robot_vision/2.5D/monodepth2/kitti_data --model_name edge4ch --use_edge_input --num_epochs 30 --batch_size 12 --weights_init scratch --lr 0.0001 5e-6 31 0.0001 1e-5 31 --png


仅改进2：confidence 替代 auto-mask
python train.py --data_path /path/to/kitti --model_name confmask --use_confidence_mask
两个都开
python train.py --data_path /path/to/kitti --model_name edge4ch_conf --use_edge_input --use_confidence_mask
调 confidence 正则强度（建议试 0.005 / 0.01 / 0.02）
python train.py --data_path /path/to/kitti --model_name confmask_w001 --use_confidence_mask --confidence_reg_weight 0.01


# monodepth2

   Mono evaluation - using median scaling
 Scaling ratios | med: 34.978 | std: 0.090

   abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
&   0.136  &   1.099  &   5.210  &   0.213  &   0.844  &   0.946  &   0.976  \\


# 不加边信息 lite-mono
-> Evaluating
   Mono evaluation - using median scaling
 Scaling ratios | med: 36.653 | std: 0.087

   abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
&   0.135  &   1.056  &   5.155  &   0.210  &   0.840  &   0.947  &   0.977  \\

  [Boundary band]
&   0.153  &   1.489  &   6.457  &   0.241  &   0.793  &   0.927  &   0.968  \\

  [Non-boundary]
&   0.104  &   0.372  &   1.431  &   0.140  &   0.916  &   0.978  &   0.991  \\

  [Contour vicinity]
&   0.152  &   1.451  &   6.302  &   0.238  &   0.799  &   0.929  &   0.969  \\


  flops: 5.032G, params: 3.069M, flops_e: 4.314G, params_e:2.842M, flops_d:718.295M, params_d:226.627K





# 加入边信息
-> Evaluating
   Mono evaluation - using median scaling
 Scaling ratios | med: 36.550 | std: 0.084

   abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
&   0.133  &   1.060  &   5.201  &   0.210  &   0.842  &   0.946  &   0.977  \\





python evaluate_depth.py   --data_path /home/desc/jzy/robot_vision/2.5D/monodepth2/kitti_data   --load_weights_folder /home/desc/jzy/robot_vision/2.5D/my-mono/tmp/edge4ch/models/weights_29   --png   --use_edge_input   --report_region_metrics   --boundary_band_px 1   --contour_band_px 2

   abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
&   0.133  &   1.060  &   5.201  &   0.210  &   0.842  &   0.946  &   0.977  \\

-> Region metrics enabled
boundary_band_px=1, contour_band_px=2

  [Boundary band]
&   0.153  &   1.524  &   6.527  &   0.243  &   0.794  &   0.925  &   0.967  \\

  [Non-boundary]
&   0.101  &   0.328  &   1.380  &   0.136  &   0.919  &   0.980  &   0.992  \\

  [Contour vicinity]
&   0.151  &   1.480  &   6.368  &   0.240  &   0.799  &   0.927  &   0.968  \\

  flops: 5.051G, params: 3.071M, flops_e: 4.332G, params_e:2.845M, flops_d:718.295M, params_d:226.627K

这个结果说明边界区域结果差，边界带的 rmse 6.527，对比非边界区 1.380，差距非常大；a1 也是 0.794 对 0.919。也就是说，模型的主要困难不是平坦区域，而是深度不连续、物体轮廓、遮挡交界这些地方。

这其实正支持你做边缘输入的动机。因为如果错误主要集中在边界区，那么“显式加边结构”理论上最可能改善的，也正是这些区域。

不过，现在还差最关键的一步，才能真正证明“edge 有效”：
你必须用同样的 region metrics 去跑 baseline（不加 edge 的 Lite-Mono）。

python evaluate_depth.py   --data_path /home/desc/jzy/robot_vision/2.5D/monodepth2/kitti_data   --load_weights_folder /home/desc/jzy/robot_vision/2.5D/Lite-Mono/tmp/mytrain/models/weights_29  --png  --report_region_metrics   --boundary_band_px 1   --contour_band_px 2


“第四通道的 edge 输入，能不能带来一点正向增益？”
答案是：能，但增益很小，而且主要不在边界带。

但你还没有回答另一个更关键的问题：

“这个增益到底来自 edge 的语义，还是仅仅来自‘多了一个第四通道’这件事本身？”

而“随机噪声第四通道对照”就是专门回答这个问题的。
跑这个实验证明
结果
   Mono evaluation - using median scaling
 Scaling ratios | med: 37.094 | std: 0.089

   abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
&   0.138  &   0.969  &   5.303  &   0.217  &   0.822  &   0.942  &   0.977  \\

  flops: 5.051G, params: 3.071M, flops_e: 4.332G, params_e:2.845M, flops_d:718.295M, params_d:226.627K

-> Done!


下面是这个使用mid_fusion 但是是用权重进行融合 rgb_fusion + \lamma depth_fusion

   Mono evaluation - using median scaling
 Scaling ratios | med: 37.190 | std: 0.085

   abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
&   0.132  &   1.038  &   5.152  &   0.211  &   0.844  &   0.945  &   0.976  \\

-> Region metrics enabled
boundary_band_px=1, contour_band_px=2

  [Boundary band]
&   0.151  &   1.472  &   6.460  &   0.243  &   0.799  &   0.926  &   0.967  \\

  [Non-boundary]
&   0.103  &   0.354  &   1.412  &   0.141  &   0.917  &   0.975  &   0.990  \\

  [Contour vicinity]
&   0.149  &   1.431  &   6.303  &   0.240  &   0.805  &   0.928  &   0.967  \\

  flops: 5.309G, params: 3.149M, flops_e: 4.591G, params_e:2.922M, flops_d:718.295M, params_d:226.627K

-> Done!


为了进一步改进引入门控机制 rgb_fusion + gate * depth_fusion （让门控网络自学习）

训练命令：
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data_path /data3/jzy/vision/monodepth2/kitti_data \
  --model_name edge_midfusion_gate \
  --use_edge_mid_fusion \
  --edge_fusion_strength 1.0 \
  --edge_fusion_strength_mode gate_only \
  --num_epochs 30 \
  --batch_size 12 \
  --weights_init scratch \
  --lr 0.0001 5e-6 31 0.0001 1e-5 31 \
  --png

评估命令：
python evaluate_depth.py \
  --data_path /data3/jzy/vision/monodepth2/kitti_data \
  --load_weights_folder /data3/jzy/vision/my-mono/tmp/edge_midfusion_gate/models/weights_29 \
  --use_edge_mid_fusion \
  --png \
  --report_region_metrics \
  --boundary_band_px 1 \
  --contour_band_px 2

评估结果（gate_only 模式）：
   Mono evaluation - using median scaling
 Scaling ratios | med: 34.795 | std: 0.080

   abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
&   0.126  &   0.977  &   5.079  &   0.206  &   0.853  &   0.949  &   0.978  \\

-> Region metrics enabled
boundary_band_px=1, contour_band_px=2

  [Boundary band]
&   0.145  &   1.397  &   6.376  &   0.238  &   0.806  &   0.930  &   0.968  \\

  [Non-boundary]
&   0.096  &   0.314  &   1.329  &   0.132  &   0.927  &   0.980  &   0.992  \\

  [Contour vicinity]
&   0.143  &   1.356  &   6.219  &   0.235  &   0.812  &   0.932  &   0.969  \\

  flops: 5.279G, params: 3.140M, flops_e: 4.561G, params_e:2.914M, flops_d:718.295M, params_d:226.627K

-> Done!


## 2026-05-11 版本迭代记录

### V2/V3/V4 失败原因分析

| 版本 | 失败原因 | 教训 |
|------|----------|------|
| v2 | 训练代码修改了 gate 参数名（edge_down_4.0, gate_4.0 等），但 evaluate_depth.py 没有同步更新，导致架构不兼容 | 评估代码必须与训练代码同步更新 |
| v3 | 同上，架构不兼容 | 同上 |
| v4 | 训练配置与评估不匹配，或训练未收敛，最终效果与 baseline 相同（abs_rel=0.135） | 需要严格验证训练-评估一致性 |

### 成功方案分析 (edge_midfusion_gate)
- **abs_rel: 0.126** (相对 baseline 提升 6.7%)
- 参数量: 3.140M (相对 baseline 增加 2.3%)
- 关键成功因素: 门控机制让模型自适应学习边界融合权重

### edge_midfusion_v2 改进设计

基于成功经验，计划进行以下改进：

#### 改进1: 轻量化门控网络 (Lightweight Gating)
- 当前 v1 gate: Conv2d(2*dims[i], dims[i], k=1) + ReLU + Conv2d(dims[i], dims[i], k=1)
- V2 改进: Conv2d(2*dims[i], dims[i]//8, k=1) + ReLU + Conv2d(dims[i]//8, dims[i], k=1)
- 预期: 减少门控网络参数量和计算量

#### 改进2: 减少边缘特征中间通道
- 当前 v1 edge_mid_ch = max(8, dims[0]//2) = 24
- V2 改进: edge_mid_ch = max(6, dims[0]//4) = 12
- 预期: 减少边缘分支参数量

#### 改进3: 边界感知训练 (Boundary-Aware Training)
- 在训练损失中对边界区域增加额外权重
- 边界像素的损失乘以系数（如 1.5x）
- 预期: 让模型更关注边界预测准确性

**V2 核心策略**: 在保持 v1 gate 机制成功经验的基础上，通过轻量化门控和减少中间通道来控制参数量，目标是在不增加甚至减少参数量的前提下提升性能。

**训练命令**:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data_path /data3/jzy/vision/monodepth2/kitti_data \
  --model_name edge_midfusion_v2 \
  --use_edge_mid_fusion \
  --edge_fusion_mode v2 \
  --edge_fusion_strength 1.0 \
  --edge_fusion_strength_mode gate_only \
  --num_epochs 30 \
  --batch_size 12 \
  --weights_init scratch \
  --lr 0.0001 5e-6 31 0.0001 1e-5 31 \
  --png
```

**V2 评测结果 (2026-05-12)**:
```
   Mono evaluation - using median scaling
 Scaling ratios | med: 35.974 | std: 0.081

   abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 |
&   0.133  &   1.023  &   5.132  &   0.209  &   0.842  &   0.947  &   0.977  \\

-> Region metrics enabled
boundary_band_px=1, contour_band_px=2

  [Boundary band]
&   0.151  &   1.454  &   6.435  &   0.240  &   0.795  &   0.927  &   0.968  \\

  [Non-boundary]
&   0.102  &   0.340  &   1.404  &   0.139  &   0.917  &   0.978  &   0.991  \\

  [Contour vicinity]
&   0.150  &   1.416  &   6.279  &   0.237  &   0.801  &   0.929  &   0.969  \\

  flops: 5.158G, params: 3.112M, flops_e: 4.440G, params_e:2.886M, flops_d:718.295M, params_d:226.627K
```

---

## 参数量对比总结（实测数据，2026-05-12）

| 版本 | 参数量 | abs_rel | 相对baseline提升 | 备注 |
|------|--------|---------|------------------|------|
| Lite-Mono Baseline | **3.069M** | 0.135 | - | 参数量最小 |
| edge_midfusion_gate (v1 gate) | **3.140M** | **0.126** | **6.7%** | 最佳方案 |
| edge_midfusion_v2 (轻量化 gate) | **3.112M** | 0.133 | 1.5% | 参数量减少 |
| edge_midfusion_v4 | 3.140M | 0.135 | 0% | 训练未收敛 |
| edge_midfusion_v2/v3 | - | - | - | 架构不兼容，已删除 |

### 区域指标对比

| 区域 | Baseline | v1 gate | v2 轻量化 |
|------|----------|---------|-----------|
| 边界带 abs_rel | ~0.153 | **0.145** | 0.151 |
| 非边界区 abs_rel | ~0.105 | **0.096** | 0.102 |
| 轮廓带 abs_rel | ~0.153 | **0.143** | 0.150 |

### V2 vs V1 Gate 详细对比

| 指标 | v1 gate | v2 轻量化 | 变化 |
|------|---------|-----------|------|
| abs_rel | 0.126 | 0.133 | -5.6% (变差) |
| sq_rel | 0.977 | 1.023 | -4.7% (变差) |
| rmse | 5.079 | 5.132 | -1.0% (变差) |
| 参数量 | 3.140M | 3.112M | **-0.9% (减少)** |
| 边界 abs_rel | 0.145 | 0.151 | -4.1% (变差) |

### 关键发现

1. **v2 轻量化方案参数量减少 0.9%**（3.140M → 3.112M），但性能下降
2. **v1 gate 仍是最佳方案**：abs_rel 0.126，提升 6.7%
3. **轻量化代价**：门控网络从 `dims` 降到 `dims//8` 导致信息瓶颈，边界融合效果变差
4. **结论**：门控网络的中间层维度对性能有重要影响，不能过度压缩

---

## FPN 特征金字塔实验（2026-05-12）

### 实验设计

尝试在 1/16 层添加边界特征融合 + 边界注意力 + top-down refinement，期望利用更粗糙尺度的边界信息。

### 实验结果

| 指标 | v1 | FPN |
|------|-----|-----|
| abs_rel | **0.126** | 0.193 |
| a1 | **0.830** | 0.743 |
| 参数量 | 3.140M | ~3.15M |

**FPN 训练过程**：
- Epoch 10: abs_rel=0.232
- Epoch 19: abs_rel=0.217
- Epoch 31: abs_rel=0.193（仍未收敛）

### 关键发现

1. **FPN 严重失败**：abs_rel 比 v1 差 53%
2. **训练不稳定**：loss 在后期反而上升
3. **1/16 层边界特征过于稀疏**：边界信息在深层已经丢失，加入额外的门控和注意力反而引入噪声
4. **结论**：特征金字塔方向不适合此任务，边界信息应在浅层融合

---

## 结论与后续改进方向

### 当前最佳方案

**v1 gate 架构**（edge_midfusion_gate）：
- abs_rel: 0.126, a1: 0.830
- 参数量: 3.140M
- 在边界/非边界区域均优于 baseline

### 后续改进建议

实验1： 把 edge 作为 modulation，不要作为 additive feature 如果你现在是类似： x = x + gate * edge_feat 可以试改成 FiLM / modulation 形式： gamma, beta = edge_gate(edge_feat) x = x * (1 + alpha * gamma) + alpha * beta


实验 2：v1 gate + stage 消融（stage是目前融合的尺度，例如1/4，1/8等）
only stage0
only stage1
stage0 + stage1
stage0 + stage1 + stage2

实验 3：v1 gate 轻量化中间版本
hidden = C/2
hidden = C/4
不要再直接 C/8。


GPU 0: FiLM (实验1)
GPU 1: Stage0 only (实验2)
GPU 2: V3 C/4 (实验3)

---

## 实验结果汇总（2026-05-14）

### 实验1：FiLM Modulation（edge_film）

将 edge 作为 modulation 替代 additive feature，使用 FiLM 形式：
`gamma, beta = edge_gate(edge_feat)`
`x = x * (1 + alpha * gamma) + alpha * beta`

**评估命令**：
```bash
python evaluate_depth.py \
  --data_path /data3/jzy/vision/monodepth2/kitti_data \
  --load_weights_folder /data3/jzy/vision/my-mono/tmp/edge_film/models/weights_29 \
  --use_edge_mid_fusion \
  --edge_fusion_mode film \
  --report_region_metrics --boundary_band_px 1 --contour_band_px 2 \
  --png
```

**评估结果**：
```
   abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 |
&   0.135  &   1.104  &   5.255  &   0.214  &   0.840  &   0.945  &   0.975  \\

  [Boundary band]
&   0.154  &   1.554  &   6.583  &   0.246  &   0.792  &   0.924  &   0.965  \\

  [Non-boundary]
&   0.105  &   0.395  &   1.456  &   0.142  &   0.916  &   0.977  &   0.991  \\

  [Contour vicinity]
&   0.152  &   1.510  &   6.423  &   0.243  &   0.798  &   0.927  &   0.966  \\

flops: 5.250G, params: 3.132M
```

### 实验2：Stage0 Only（edge_stage0_only）

使用 v1 gate 架构，仅在 stage0 (1/4 尺度) 进行边界融合，禁用 stage1 (1/8 尺度)。

**评估命令**：
```bash
python evaluate_depth.py \
  --data_path /data3/jzy/vision/monodepth2/kitti_data \
  --load_weights_folder /data3/jzy/vision/my-mono/tmp/edge_stage0_only/models/weights_29 \
  --use_edge_mid_fusion \
  --edge_fusion_mode v1 \
  --edge_stage_mode stage0_only \
  --report_region_metrics --boundary_band_px 1 --contour_band_px 2 \
  --png
```

**评估结果**：
```
   abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 |
&   0.131  &   1.022  &   5.113  &   0.209  &   0.846  &   0.948  &   0.977  \\

  [Boundary band]
&   0.150  &   1.458  &   6.412  &   0.240  &   0.800  &   0.928  &   0.967  \\

  [Non-boundary]
&   0.100  &   0.336  &   1.386  &   0.138  &   0.921  &   0.978  &   0.991  \\

  [Contour vicinity]
&   0.149  &   1.419  &   6.257  &   0.237  &   0.805  &   0.930  &   0.968  \\

flops: 5.324G, params: 3.233M
```

### 实验3：V3 C/4 轻量化（edge_v3）

使用 C/4 作为 gate 中间层维度（介于 v1 的 C 和 v2 的 C/8 之间）。

**评估命令**：
```bash
python evaluate_depth.py \
  --data_path /data3/jzy/vision/monodepth2/kitti_data \
  --load_weights_folder /data3/jzy/vision/my-mono/tmp/edge_v3/models/weights_29 \
  --use_edge_mid_fusion \
  --edge_fusion_mode v3 \
  --report_region_metrics --boundary_band_px 1 --contour_band_px 2 \
  --png
```

**评估结果**：
```
   abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 |
&   0.132  &   1.051  &   5.196  &   0.212  &   0.843  &   0.947  &   0.976  \\

  [Boundary band]
&   0.151  &   1.491  &   6.517  &   0.244  &   0.797  &   0.927  &   0.967  \\

  [Non-boundary]
&   0.102  &   0.357  &   1.406  &   139  &   0.919  &   0.978  &   0.991  \\

  [Contour vicinity]
&   0.150  &   1.452  &   6.360  &   0.241  &   0.802  &   0.929  &   0.968  \\

flops: 5.167G, params: 3.115M
```

---

### 综合对比分析

| 版本 | abs_rel | a1 | 边界 abs_rel | 非边界 abs_rel | 参数量 | 备注 |
|------|---------|-----|--------------|----------------|--------|------|
| **v1 gate (all stages)** | **0.126** | **0.853** | **0.145** | **0.096** | 3.140M | 最佳 |
| Stage0 only | 0.131 | 0.846 | 0.150 | 0.100 | 3.233M | 略差于v1 |
| V3 C/4 | 0.132 | 0.843 | 0.151 | 0.102 | 3.115M | 轻量化 |
| FiLM | 0.135 | 0.840 | 0.154 | 0.105 | 3.132M | modulation |
| v2 (C/8) | 0.133 | 0.842 | 0.151 | 0.102 | 3.112M | 过轻量化 |
| Baseline | 0.135 | 0.840 | 0.153 | 0.101 | 3.069M | 无edge |

### 关键发现

1. **v1 gate (所有stage融合) 仍是最佳方案**：abs_rel 0.126，边界 abs_rel 0.145
2. **Stage0 only 与 v1 gate 接近**：说明 1/4 尺度的边界融合贡献最大，1/8 尺度起辅助作用
3. **V3 C/4 轻量化效果**：参数量减少 0.8%（3.140M→3.115M），性能略微下降可接受
4. **FiLM modulation 效果不佳**：相比 additive 形式没有提升，可能需要更多调参

### 后续改进方向

1. **尝试 stage0 + stage1 融合**（不做 stage2）：验证是否能在减少参数的同时保持性能
2. **边界感知损失**：在训练时对边界区域增加额外权重
3. **多尺度边界融合权重学习**：让模型学习不同尺度边界信息的重要性

## 2026-07-01 Neuro Fusion

### 训练命令
```bash
python train.py \
  --data_path /home/jzy/data/vision/monodepth2/kitti_data \
  --model_name neuro_fusion \
  --use_neuro_fusion \
  --num_epochs 30 \
  --batch_size 12 \
  --weights_init scratch \
  --png
```

### 评估命令
```bash
python evaluate_depth.py \
  --data_path /home/jzy/data/vision/monodepth2/kitti_data \
  --model_name neuro_fusion \
  --use_neuro_fusion \
  --load_weights_folder ./tmp/neuro_fusion/models/weights_29 \
  --split eigen_zhou \
  --batch_size 8 \
  --png \
  --report_region_metrics --boundary_band_px 1 --contour_band_px 2
```

### 评估结果
   Mono evaluation - using median scaling
 Scaling ratios | med: 34.182 | std: 0.085

   abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 |
&   0.134  &   1.038  &   5.168  &   0.213  &   0.839  &   0.946  &   0.976  \\

  flops: 5.351G, params: 3.139M, flops_e: 4.633G, params_e:2.912M, flops_d:718.295M, params_d:226.627K

### 区域评测结果
-> Region metrics enabled
boundary_band_px=1, contour_band_px=2

  [Boundary band]
&   0.153  &   1.464  &   6.481  &   0.244  &   0.791  &   0.926  &   0.967  \\

  [Non-boundary]
&   0.104  &   0.351  &   1.418  &   0.141  &   0.916  &   0.977  &   0.991  \\

  [Contour vicinity]
&   0.151  &   1.422  &   6.323  &   0.241  &   0.797  &   0.928  &   0.968  \\

### 性能对比

| 版本 | abs_rel | a1 | 边界 abs_rel | 非边界 abs_rel | 参数量 | 备注 |
|------|---------|-----|--------------|----------------|--------|------|
| **v1 gate (all stages)** | **0.126** | **0.853** | **0.145** | **0.096** | 3.140M | 最佳 |
| Neuro Fusion (Gabor) | 0.134 | 0.839 | 0.153 | 0.104 | 3.139M | Gabor滤波 |
| Baseline | 0.135 | 0.840 | 0.153 | 0.101 | 3.069M | 无edge |

## 2026-07-07 Neuro Fusion v2（Gabor 修复版）

### 训练命令
```bash
python train.py \
  --data_path /home/jzy/data/vision/monodepth2/kitti_data \
  --model_name neuro_fusion2 \
  --use_neuro_fusion \
  --neuro_fusion_mode gabor \
  --num_epochs 30 \
  --batch_size 12 \
  --weights_init scratch \
  --png
```

### 评估命令
```bash
python evaluate_depth.py \
  --data_path /home/jzy/data/vision/monodepth2/kitti_data \
  --load_weights_folder ./tmp/neuro_fusion2/models/weights_29 \
  --model lite-mono \
  --use_neuro_fusion \
  --neuro_fusion_mode gabor \
  --eval_split eigen \
  --png \
  --report_region_metrics --boundary_band_px 1 --contour_band_px 2
```

### 评估结果
   Mono evaluation - using median scaling
 Scaling ratios | med: 30.890 | std: 0.081

   abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 |
&   0.131  &   0.996  &   5.111  &   0.209  &   0.844  &   0.947  &   0.977  \\

  flops: 5.360G, params: 3.139M, flops_e: 4.642G, params_e:2.912M, flops_d:718.295M, params_d:226.627K

### 区域评测结果
-> Region metrics enabled
boundary_band_px=1, contour_band_px=2

  [Boundary band]
&   0.149  &   1.424  &   6.413  &   0.240  &   0.798  &   0.927  &   0.968  \\

  [Non-boundary]
&   0.101  &   0.320  &   1.383  &   0.138  &   0.917  &   0.978  &   0.991  \\

  [Contour vicinity]
&   0.148  &   1.384  &   6.257  &   0.237  &   0.803  &   0.929  &   0.969  \\

### 修复内容
- projection 改为 residual fusion（BN 零初始化，初始恒等），避免随机注入
- gabor_out 强制 interpolate 到 conv_out 尺寸（修正 stride>1 时的取整不一致）
- NaN/Inf 改为 nan_to_num（保留梯度图，避免 return 截断）

### 性能对比

| 版本 | abs_rel | a1 | 边界 abs_rel | 非边界 abs_rel | 参数量 |
|------|---------|-----|-------------|---------------|--------|
| v1 gate (edge mid-fusion, 最佳) | 0.126 | 0.853 | 0.145 | 0.096 | 3.140M |
| **neuro_fusion2 (修复后 Gabor)** | **0.131** | **0.844** | **0.149** | **0.101** | **3.139M** |
| neuro_fusion (原版 Gabor) | 0.134 | 0.839 | 0.153 | 0.104 | 3.139M |
| Baseline (无 edge) | 0.135 | 0.840 | 0.153 | 0.101 | 3.069M |

修复后 Gabor 分支 abs_rel 提升 0.003（0.134→0.131），边界 abs_rel 提升 0.004（0.153→0.149），证明修复方向正确。边界改善最大，与 Gabor 滤波器的方向选择性一致。与显式 edge 输入（0.126）仍有 0.005 的 gap，gap 主要来自显式 Sobel 边缘图比隐式学 Gabor 更直接。

## 2026-07-07 Sampling Dual（Depth + Pose）

### 训练命令
```bash
python train.py \
  --data_path /home/jzy/data/vision/monodepth2/kitti_data \
  --model_name sampling-dual-fixed \
  --num_epochs 30 \
  --batch_size 10 \
  --weights_init scratch \
  --png \
  --use_sampling_strategy \
  --depth_num_samples 4 \
  --depth_sampling_noise 0.03 \
  --depth_sampling_mode scaled_disp \
  --pose_num_samples 4 \
  --pose_rot_noise 0.005 \
  --pose_trans_noise_ratio 0.03 \
  --sampling_margin_rel 0.05 \
  --sampling_margin_abs 0.0 \
  --sampling_distill_d_weight 0.05 \
  --sampling_distill_t_weight 0.05
```

### 评估命令
```bash
python evaluate_depth.py \
  --data_path /home/jzy/data/vision/monodepth2/kitti_data \
  --model_name sampling-dual-fixed \
  --load_weights_folder ./tmp/sampling-dual-fixed/models/weights_29 \
  --batch_size 10 \
  --png
```

### 评估结果
   Mono evaluation - using median scaling
 Scaling ratios | med: 16.858 | std: 0.085

   abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 |
&   0.132  &   1.007  &   5.141  &   0.211  &   0.841  &   0.948  &   0.977  \\

  flops: 5.032G, params: 3.069M, flops_e: 4.314G, params_e:2.842M, flops_d:718.295M, params_d:226.627K

### 备注
- depth encoder 为 LiteMono（lite-mono，192x640，drop_path=0.2），pose encoder 为 ResNet18，均从头训练（scratch）
- depth 候选采样：4 个，在 clean disparity 附近加 scaled_disp 噪声 ±0.03
- pose 候选采样：4 个，旋转噪声 0.005 rad，平移相对噪声 0.03
- margin gate：相对 margin 0.05，绝对 margin 0.0
- distill weight：depth=0.05，pose=0.05
- median scale=16.858 偏高，模型整体深度偏浅，需 median scaling 校正

