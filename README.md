# NGF-Mono

**Neuro-inspired Gated Frequency Fusion for Self-Supervised Monocular Depth Estimation**

**神经启发式门控频率融合的自监督单目深度估计**

NGF-Mono 是本项目采用的方法名，核心是在 Lite-Mono 编码器旁增加可学习 Gabor 高频分支，通过门控残差融合补充图像结构信息。仓库还保留 Sobel 边缘输入与融合、置信度掩码、几何一致性以及深度／姿态采样等实验，用于对照和消融。

本项目基于 Lite-Mono 扩展，复用其 CNN 与 Transformer 主干、深度解码器及自监督训练框架。方法名描述本项目的扩展，不代表已发表论文；代码中的 `LiteMono`、`--model lite-mono` 等标识仍保留。

## 目录

- [方法与实现](#方法与实现)
- [与 Lite-Mono 的隔离检查](#与-lite-mono-的隔离检查)
- [环境准备](#环境准备)
- [数据准备](#数据准备)
- [训练](#训练)
- [评估与推理](#评估与推理)
- [本地实验记录](#本地实验记录)
- [目录结构](#目录结构)
- [已知限制](#已知限制)
- [致谢与引用](#致谢与引用)

## 方法与实现

### 高频分支与门控融合

主干提取多尺度特征；独立高频分支接收归一化后的 RGB 图像，通过可学习 Gabor 滤波器提取方向与频率响应，并在主干的 1/4、1/8 分辨率处融合。

```text
RGB ── 归一化 ── Lite-Mono Stem ── Stage 0 ── (+) ── Stage 1 ── (+) ── Stage 2
          │                                  ↑                  ↑          │
          └── Gabor 高频分支 ── HF_4 ── 门控融合 ┘                  │          │
                                  └── HF_8 ────── 门控融合 ──────┘          │
                                     多尺度主干特征 ── DepthDecoder ── 视差
```

融合形式：

```text
F_out = F_rgb + α · sigmoid(G(concat(F_rgb, F_hf))) · F_hf
```

`α` 是可学习缩放参数，两个融合位置分别初始化为 `0.1`。Gabor 层对方向与频率响应做能量聚合。实现见 [networks/depth_encoder.py](networks/depth_encoder.py) 中的 `LearnableGaborConv2d`、`NeuroHFEncoderBranch` 和 `LiteMono.forward_features`。

### 可选实验模块

以下模块均需显式启用，默认命令不会自动运行 NGF-Mono。

| 模块 | 启用参数 | 作用 |
| --- | --- | --- |
| Gabor 高频融合，即 NGF-Mono 核心 | `--use_neuro_fusion` | 从 RGB 提取高频特征，在 1/4、1/8 尺度门控融合 |
| Sobel 第四通道 | `--use_edge_input` | 将边缘图拼接为第四通道；支持零值、噪声、打乱或模糊边缘对照 |
| Sobel 中层融合 | `--use_edge_mid_fusion` | 独立边缘分支，支持 `v1/v2/v3/v4/film` 融合形式 |
| 学习式置信度 | `--use_confidence_mask` | 预测各源帧置信度，加权重投影损失，自动关闭 auto-masking |
| 几何一致性 | `--use_geometric_loss` | 对目标帧与源帧预测深度施加几何一致性约束 |
| 深度／姿态采样 | `--use_sampling_strategy` | 在停止梯度的候选池中搜索，超过改善阈值后增加蒸馏辅助损失 |
| DINOv2 主干对照 | `--depth_encoder_type dino` | 替换深度主干；当前存在路径与参数兼容问题，见已知限制 |

默认训练通过相邻视频帧及预测位姿构造重投影监督，使用 SSIM 与 L1 光度损失、auto-masking 和边缘感知视差平滑项。几何一致性与采样蒸馏是附加项；`--geometric_loss_only` 则切换为仅几何损失。

## 与 Lite-Mono 的隔离检查

**结论：新增功能有开关隔离，尚未实现独立源码、独立运行环境或严格的原版基线隔离。** 以下结论来自本次工作区源码、默认参数和 Git 配置的静态检查。

| 检查项 | 当前情况 | 影响 |
| --- | --- | --- |
| 仓库身份 | 项目远程为 `https://github.com/jzy-doc/NGF-Mono.git`，Lite-Mono 原远程保留为 `upstream` | 已区分项目与上游的远程地址，源码仍共用 |
| 源码边界 | 新增模块直接写入 `LiteMono` 类，训练共用 `Trainer` | 没有独立 NGF-Mono 类或保留不变的原版训练入口 |
| 默认开关 | 边缘输入、边缘融合、高频融合、置信度、几何损失和采样均默认关闭 | 可运行本仓库内部的无扩展基线 |
| 原版等价性 | 与本地 Git `HEAD` 比较，九个原有编码器基础类的 AST 一致；`LiteMono` 和训练器已修改 | 关闭开关不能证明与原版训练结果严格一致，仍需同配置数值对照 |
| 实验输出 | 输出位置为 `log_dir/model_name`，默认 `./tmp/lite-mono` | 不同实验需指定不同名称，重复名称会复用输出位置 |
| 权重加载 | 评估按命令行重建结构，再过滤键和形状；编码器使用 `strict=False` | 配置不匹配时可能忽略分支权重或保留随机初始化参数 |
| 配置恢复 | 训练保存 `models/opt.json`，评估不会自动读取；编码器只额外保存高、宽和 stereo 标记 | 评估需手动恢复主干类型和融合配置 |
| 环境与缓存 | 无根目录依赖锁文件；`networks/__init__.py` 无条件导入 DINO，后者在导入时修改全局 Torch Hub 缓存目录 | 可选主干仍有全局副作用，运行环境未隔离 |
| 上游许可 | 保留目标仓库已有的 `LICENSE`，上游 MIT 许可另存为 `LICENSE-Lite-Mono` | 上游代码仍保留原作者及许可归属 |

建议将关闭扩展后的配置称为“本地 Lite-Mono 基线”。严格比较原版时，应另建上游 checkout 和虚拟环境，固定提交、依赖、初始化、数据划分及训练设置，并分别保存输出。代码隔离还需拆分扩展类、延迟导入可选主干，并校验 checkpoint 与配置；这些尚未完成。

本次核查未运行训练或 KITTI 数值评估。检查环境缺少 `torchvision`、`timm`、`thop`、`scikit-image`，因此不对运行时等价性作结论。

## 环境准备

在项目根目录建立独立环境，并安装相互兼容的 PyTorch 与 torchvision。CUDA 版本需匹配本机驱动；仓库没有经过验证的完整版本锁定清单。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install numpy Pillow opencv-python scikit-image matplotlib timm thop tensorboardX tensorboard PyYAML six
```

上述列表根据当前源码导入整理，不是已验证的环境快照。当前 `ChainedScheduler` 已定义在 [utils.py](utils.py)，无需按旧 README 安装外部调度器 Git 包。

训练和评估主要面向 NVIDIA GPU。`evaluate_depth.py` 直接调用 `.cuda()`，目前不能通过 `--no_cuda` 切换到 CPU 评估。

## 数据准备

使用 KITTI Raw 视频序列、标定文件和 Velodyne 数据。训练列表位于 `splits/eigen_zhou/`，评估列表位于 `splits/eigen/test_files.txt`。

```text
/path/to/kitti_data/
└── 2011_09_26/
    ├── calib_cam_to_cam.txt
    ├── calib_velo_to_cam.txt
    └── 2011_09_26_drive_0001_sync/
        ├── image_02/data/
        ├── image_03/data/
        └── velodyne_points/data/
```

默认读取 JPG；保留原始 PNG 时，训练和评估命令都应加 `--png`。默认视频训练加载 `[0, -1, 1]`，需要相邻帧。

Eigen 评估还需 `splits/eigen/gt_depths.npz`。本仓库包含此文件；仓库没有 `export_gt_depth.py`。替换测试列表或重建真值时，需确保二者逐项对应。

## 训练

以下命令均在项目根目录执行。替换实际数据路径，并为每个实验使用独立名称。

### NGF-Mono

```bash
python train.py \
  --data_path /path/to/kitti_data \
  --log_dir ./tmp --model_name ngf_mono \
  --model lite-mono \
  --use_neuro_fusion --neuro_fusion_mode gabor \
  --height 192 --width 640 \
  --num_epochs 30 --batch_size 12 \
  --weights_init scratch \
  --lr 0.0001 5e-6 31 0.0001 1e-5 31 \
  --png
```

`--weights_init` 控制默认独立 ResNet 位姿编码器的预训练初始化。深度编码器加载 ImageNet 权重需要另传 `--mypretrain /path/to/encoder_pretrain.pth`；只设置 `--weights_init pretrained` 不会自动加载 Lite-Mono 深度主干预训练权重。

### 本地基线与消融

基线使用上面的训练命令，删除 `--use_neuro_fusion --neuro_fusion_mode gabor`，并将实验名改为 `baseline_local`。其他消融在同一基线设置上添加以下参数，并使用独立实验名。

| 实验名示例 | 附加参数 |
| --- | --- |
| `edge4ch` | `--use_edge_input` |
| `edge4ch_gaussian` | `--use_edge_input --fourth_channel_mode gaussian` |
| `edge_midfusion_gate` | `--use_edge_mid_fusion --edge_fusion_mode v1 --edge_stage_mode all` |
| `confidence` | `--use_confidence_mask --confidence_reg_weight 0.01` |
| `geometry` | `--use_geometric_loss --geometric_loss_weight 1.0` |
| `sampling` | `--use_sampling_strategy --sampling_distill_d_weight 0.01 --sampling_distill_t_weight 0.01` |

第四通道输入不能与中层边缘融合或 Gabor 高频融合同时开启。做分支对照时，请分别启用 Sobel 或 Gabor；代码未禁止二者同时开启，但该组合不是上述默认 NGF-Mono 配置。采样会额外评估候选深度与位姿，增加训练开销。

### 多 GPU 与 YAML

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --data_path /path/to/kitti_data \
  --model_name ngf_mono_ddp \
  --model lite-mono --use_neuro_fusion \
  --height 192 --width 640 \
  --num_epochs 30 --batch_size 12 \
  --weights_init scratch --png
```

`batch_size` 是每张 GPU 的批量，示例总批量为 48。训练支持 `--config path/to/config.yaml`，显式命令行参数覆盖 YAML 默认值；该配置加载逻辑仅在 `train.py` 中实现。

现有 [configs/replica_train.yaml](configs/replica_train.yaml) 实际使用 KITTI 与几何一致性损失，并非 Replica 数据加载配置，也不是 NGF-Mono 默认配置。它包含本机路径及 `./tmp/geo_combo/models/weights_5/encoder.pth` 前置权重依赖，使用前需调整。`launch_train.sh` 默认读取此配置，GPU 检测针对 RTX 3090；其他硬件可直接使用 `torchrun`。

### 输出与可视化

```text
tmp/ngf_mono/
├── train/                     # TensorBoard 训练日志
├── val/                       # TensorBoard 验证日志
└── models/
    ├── opt.json               # 实验配置
    └── weights_29/            # epoch 从 0 开始编号
        ├── encoder.pth
        ├── depth.pth
        ├── pose_encoder.pth
        ├── pose.pth
        ├── adam.pth
        └── adam_pose.pth
```

GitHub 仓库只保留 `tmp/` 下一级实验目录的 `.gitkeep`，不包含 checkpoint、TensorBoard 日志或其他训练产物；本地已有内容仍保留。启用附加网络时会保存对应权重，例如 `confidence.pth`。查看日志：

```bash
tensorboard --logdir ./tmp/ngf_mono
```

## 评估与推理

### KITTI Eigen 评估

评估 NGF-Mono 必须显式启用与训练一致的 Gabor 分支：

```bash
python evaluate_depth.py \
  --data_path /path/to/kitti_data \
  --load_weights_folder ./tmp/ngf_mono/models/weights_29 \
  --model lite-mono \
  --use_neuro_fusion --neuro_fusion_mode gabor \
  --eval_split eigen --png \
  --report_region_metrics \
  --boundary_band_px 1 --contour_band_px 2
```

评估基线时删除高频分支参数；评估 Sobel 中层融合时，改用训练时的 `--use_edge_mid_fusion --edge_fusion_mode ... --edge_stage_mode ...`。第四通道模型也需保持通道模式一致。评估不会从文件夹名推断结构，也不会自动读取 `opt.json`。

默认采用 Eigen 裁剪、深度范围 `(1e-3, 80)` 米和逐图中值尺度对齐，报告 Abs Rel、Sq Rel、RMSE、RMSE log、δ₁、δ₂、δ₃。区域指标依据 GT 深度梯度构造边界与轮廓邻域，不是语义分割轮廓。只有协议、区域带宽和后处理设置一致时才适合比较结果。

置信度仅用于训练时，不必在普通深度评估中加载；分析置信度可在对应模型命令中增加 `--eval_confidence --report_confidence_stats`，并确保权重目录包含 `confidence.pth`。

### 单图推理

`test_simple.py` 仍按原版 RGB Lite-Mono 结构构建模型，以下命令仅适用于无附加分支的权重：

```bash
python test_simple.py \
  --load_weights_folder ./tmp/baseline_local/models/weights_29 \
  --image_path /path/to/image.jpg \
  --model lite-mono
```

该脚本没有 NGF-Mono 或 Sobel 分支参数，直接传入 Gabor 权重可能忽略高频分支，不能正确运行扩展模型。扩展模型目前应使用匹配结构的评估入口；通用单图推理入口尚待补充。单目自监督预测存在尺度不确定性，中值对齐的 KITTI 结果不等于直接恢复绝对尺度。

## 本地实验记录

下表摘自 [results.md](results.md)，用于索引历史实验，**不是本次重新评估的结果，也不是 Lite-Mono 原论文指标**。记录未完整绑定代码提交、依赖版本及随机种子，不能保证当前版本复现相同数值，或证明各行训练配置严格一致。

| 历史配置 | Abs Rel ↓ | Sq Rel ↓ | RMSE ↓ | RMSE log ↓ | δ₁ ↑ | δ₂ ↑ | δ₃ ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 本地无边缘基线 | 0.135 | 1.056 | 5.155 | 0.210 | 0.840 | 0.947 | 0.977 |
| Sobel 第四通道 | 0.133 | 1.060 | 5.201 | 0.210 | 0.842 | 0.946 | 0.977 |
| Sobel 中层融合，v1 gate | 0.126 | 0.977 | 5.079 | 0.206 | 0.853 | 0.949 | 0.978 |
| Gabor 高频融合，`neuro_fusion2` | 0.131 | 0.996 | 5.111 | 0.209 | 0.844 | 0.947 | 0.977 |

已有记录中 Sobel v1 gate 优于 Gabor 分支，因此目前不能宣称 NGF-Mono 是这些实验中的最佳模型。几何、采样及多模块组合应通过独立消融确认效果。

## 目录结构

```text
.
├── networks/                   # 主干、高频分支、DINO 适配器、深度与位姿解码器
├── datasets/                   # KITTI 数据集及图像预处理
├── splits/                     # 数据文件列表
├── configs/                    # YAML 训练配置
├── train.py                    # 单 GPU / DDP 入口及 YAML 加载
├── trainer.py                  # 网络构建、训练、损失及权重保存
├── options.py                  # 训练和评估参数
├── evaluate_depth.py           # KITTI 深度、区域及置信度评估
├── test_simple.py              # 原版结构的单图／图像目录推理
├── layers.py                   # 几何变换、SSIM、深度与姿态采样等
├── utils.py                    # 工具及学习率调度器
├── analyze_confidence.py       # 置信度分析
├── analyze_edge_usage.py       # 边缘分支分析
├── vis_neuro_hf_edge.py        # 高频与边缘特征可视化
├── generate_sampling_vis.py    # 采样可视化
├── test_replica_depth.py       # Replica 实验，需核对本机路径与结构配置
├── lite-mono-pretrain-code/    # 上游 ImageNet 主干预训练代码
├── results.md                  # 历史命令与指标
└── run.md                      # 历史运行笔记，包含本机路径
```

## 已知限制

- **部分参数没有接入计算。** Gabor 前向始终在 1/4、1/8 两处融合，没有按 `--neuro_fusion_stage` 筛选；`neuro_multiscale` 未实现额外多尺度处理。当前 Sobel 融合函数也没有使用 `edge_fusion_strength` 和 `edge_fusion_strength_mode`，不能将这些参数作为已生效的消融变量。
- **Sobel stage 命名需核对。** `edge_stage_mode=all` 实际融合 1/4、1/8；`stage0_stage1` 的前向条件却指向 1/16 路径，不符合名称直觉。本文 Sobel 示例只使用 `all`。
- **初始化注释与最终状态存在差异。** Gabor 层局部将残差投影 BatchNorm 权重置零，但外层 `LiteMono` 的 `self.apply(self._init_weights)` 会重新将其置为 1，不能据局部注释宣称最终模型保持零残差初始化。
- **DINO 存在全局副作用。** 导入 `networks/dino_encoder.py` 会将 Torch Hub 缓存设置为 `/data3/jzy/.cache/torch/hub`，即使不选择 DINO 也会执行。实现调用官方 Torch Hub，但命令行默认模型名是 timm 风格；该路径需先修正缓存位置，并显式使用匹配的模型名，如 `dinov2_vits14`。
- **脚本与环境尚未统一。** 辅助脚本、历史笔记和配置含本机路径，根目录没有锁定依赖；新增模块仍共用上游实现，严格复现与发布前需要进一步整理。

## 致谢与引用

本项目基于 Lite-Mono 开发，并沿用 Monodepth2 风格的自监督训练框架。使用时请保留上游工作归属及适用的许可文件。目标仓库原有的 Apache 2.0 许可保留于 [LICENSE](LICENSE)，Lite-Mono 上游 MIT 许可保留于 [LICENSE-Lite-Mono](LICENSE-Lite-Mono)。

上游代码与论文资料：

- [Lite-Mono 代码仓库](https://github.com/noahzn/Lite-Mono)
- [Lite-Mono 论文](https://arxiv.org/abs/2211.13202)
- [Monodepth2 代码仓库](https://github.com/nianticlabs/monodepth2)

```bibtex
@InProceedings{Zhang_2023_CVPR,
    author    = {Zhang, Ning and Nex, Francesco and Vosselman, George and Kerle, Norman},
    title     = {Lite-Mono: A Lightweight CNN and Transformer Architecture for Self-Supervised Monocular Depth Estimation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {18537-18546}
}
```
