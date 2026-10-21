from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class LiteMonoOptions:
     def __init__(self):
          self.parser = argparse.ArgumentParser(description="Lite-Mono options")

          # PATHS
          self.parser.add_argument("--data_path",
                                   type=str,
                                   help="path to the training data",
                                   default=os.path.join(file_dir, "kitti_data"))
          self.parser.add_argument("--log_dir",
                                   type=str,
                                   help="log directory",
                                   default="./tmp")

          # TRAINING options
          self.parser.add_argument("--model_name",
                                   type=str,
                                   help="the name of the folder to save the model in",
                                   default="lite-mono")
          self.parser.add_argument("--split",
                                   type=str,
                                   help="which training split to use",
                                   choices=["eigen_zhou", "eigen_full", "odom", "benchmark"],
                                   default="eigen_zhou")
          self.parser.add_argument("--model",
                                   type=str,
                                   help="which model to load",
                                   choices=["lite-mono", "lite-mono-small", "lite-mono-tiny", "lite-mono-8m"],
                                   default="lite-mono")
          self.parser.add_argument("--depth_encoder_type",
                                   type=str,
                                   help="depth encoder backbone type",
                                   choices=["lite_mono", "dino"],
                                   default="lite_mono")
          self.parser.add_argument("--dino_model_name",
                                   type=str,
                                   help="timm DINO model name when depth_encoder_type=dino",
                                   default="vit_small_patch14_dinov2.lvd142m")
          self.parser.add_argument("--dino_pretrained",
                                   help="if set, use pretrained DINO backbone",
                                   action="store_true")
          self.parser.add_argument("--freeze_dino_backbone",
                                   type=bool,
                                   help="if set, freeze DINO backbone and only train adapter/decoder",
                                   default=True)
          self.parser.add_argument("--weight_decay",
                                   type=float,
                                   help="weight decay in AdamW",
                                   default=1e-2)
          self.parser.add_argument("--drop_path",
                                   type=float,
                                   help="drop path rate",
                                   default=0.2)
          self.parser.add_argument("--num_layers",
                                   type=int,
                                   help="number of resnet layers",
                                   default=18,
                                   choices=[18, 34, 50, 101, 152])
          self.parser.add_argument("--dataset",
                                   type=str,
                                   help="dataset to train on",
                                   default="kitti",
                                   choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test"])
          self.parser.add_argument("--png",
                                   help="if set, trains from raw KITTI png files (instead of jpgs)",
                                   action="store_true")
          self.parser.add_argument("--height",
                                   type=int,
                                   help="input image height",
                                   default=192)
          self.parser.add_argument("--width",
                                   type=int,
                                   help="input image width",
                                   default=640)
          self.parser.add_argument("--disparity_smoothness",
                                   type=float,
                                   help="disparity smoothness weight",
                                   default=1e-3)
          self.parser.add_argument("--scales",
                                   nargs="+",
                                   type=int,
                                   help="scales used in the loss",
                                   default=[0, 1, 2])
          self.parser.add_argument("--min_depth",
                                   type=float,
                                   help="minimum depth",
                                   default=0.1)
          self.parser.add_argument("--max_depth",
                                   type=float,
                                   help="maximum depth",
                                   default=100.0)
          self.parser.add_argument("--use_stereo",
                                   help="if set, uses stereo pair for training",
                                   action="store_true")
          self.parser.add_argument("--frame_ids",
                                   nargs="+",
                                   type=int,
                                   help="frames to load",
                                   default=[0, -1, 1])

          self.parser.add_argument("--profile",
                                   type=bool,
                                   help="profile once at the beginning of the training",
                                   default=True)

          # OPTIMIZATION options
          self.parser.add_argument("--batch_size",
                                   type=int,
                                   help="batch size",
                                   default=16)
          self.parser.add_argument("--lr",
                                   nargs="+",
                                   type=float,
                                   help="learning rates of DepthNet and PoseNet. "
                                        "Initial learning rate, "
                                        "minimum learning rate, "
                                        "First cycle step size.",
                                   default=[0.0001, 5e-6, 31, 0.0001, 1e-5, 31])
          self.parser.add_argument("--num_epochs",
                                   type=int,
                                   help="number of epochs",
                                   default=50)
          self.parser.add_argument("--start_epoch",
                                   type=int,
                                   help="starting epoch for resumed training (default 0)",
                                   default=0)
          self.parser.add_argument("--scheduler_step_size",
                                   type=int,
                                   help="step size of the scheduler",
                                   default=15)

          # ABLATION options
          self.parser.add_argument("--v1_multiscale",
                                   help="if set, uses monodepth v1 multiscale",
                                   action="store_true")
          self.parser.add_argument("--avg_reprojection",
                                   help="if set, uses average reprojection loss",
                                   action="store_true")
          self.parser.add_argument("--disable_automasking",
                                   help="if set, doesn't do auto-masking",
                                   action="store_true")
          self.parser.add_argument("--predictive_mask",
                                   help="if set, uses a predictive masking scheme as in Zhou et al",
                                   action="store_true")
          self.parser.add_argument("--no_ssim",
                                   help="if set, disables ssim in the loss",
                                   action="store_true")
          self.parser.add_argument("--use_edge_input",
                                   help="if set, append Sobel edge map as 4th channel for encoder input",
                                   action="store_true")
          self.parser.add_argument("--use_edge_mid_fusion",
                                   help="if set, use edge branch with gated middle-scale fusion (1/4 and 1/8)",
                                   action="store_true")
          self.parser.add_argument("--edge_midfusion_gaussian_noise",
                                   help="if set, use gaussian noise instead of sobel edge for midfusion",
                                   action="store_true")
          self.parser.add_argument("--edge_fusion_strength",
                                   type=float,
                                   help="global scaling factor for gated edge fusion residual",
                                   default=1.0)
          self.parser.add_argument("--edge_fusion_strength_mode",
                                   type=str,
                                   help="fusion residual scaling mode for edge_mid_fusion",
                                   default="gate_only",
                                   choices=["fixed", "gate_only"])
          self.parser.add_argument("--edge_fusion_mode",
                                   type=str,
                                   help="edge fusion mode: v1 (original gate), v2 (lightweight C/8), v3 (C/4), v4 (C/2), film (FiLM modulation)",
                                   default="v1",
                                   choices=["v1", "v2", "v3", "v4", "film"])
          self.parser.add_argument("--edge_stage_mode",
                                   type=str,
                                   help="which stages to fuse: all (1/4+1/8), stage0_only, stage1_only, stage0_stage1",
                                   default="all",
                                   choices=["all", "stage0_only", "stage1_only", "stage0_stage1"])
          
          # =====================================================================
          # Neuro-inspired Implicit High-Frequency Fusion Options
          # 神经启发式隐式高频融合选项
          # =====================================================================
          # This replaces hand-crafted Sobel edge detection with learned
          # neuro-inspired filters (Gabor-like, Simple/Complex cells)
          # =====================================================================
          self.parser.add_argument("--use_neuro_fusion",
                                   help="if set, use neuro-inspired implicit high-frequency feature fusion",
                                   action="store_true")
          self.parser.add_argument("--neuro_fusion_mode",
                                   type=str,
                                   help="neuro fusion mode: gabor (learnable Gabor filters)",
                                   default="gabor",
                                   choices=["gabor"])
          self.parser.add_argument("--neuro_fusion_stage",
                                   type=str,
                                   help="which stages to apply neuro fusion: all, stage0_only, stage1_only",
                                   default="all",
                                   choices=["all", "stage0_only", "stage1_only", "stage0_stage1"])
          self.parser.add_argument("--neuro_multiscale",
                                   help="if set, use multi-scale neuro extraction",
                                   action="store_true")
          self.parser.add_argument("--edge_epsilon",
                                   type=float,
                                   help="numerical stability epsilon for Sobel magnitude normalization",
                                   default=1e-6)
          self.parser.add_argument("--fourth_channel_mode",
                                   type=str,
                                   help="4th channel mode when use_edge_input is enabled",
                                   default="edge",
                                   choices=["edge", "zero", "gaussian", "shuffled_edge", "blurred_edge"])
          self.parser.add_argument("--fourth_channel_noise_std",
                                   type=float,
                                   help="std for gaussian 4th-channel mode",
                                   default=0.2)
          self.parser.add_argument("--fourth_channel_blur_kernel",
                                   type=int,
                                   help="odd kernel size for blurred_edge mode",
                                   default=5)
          self.parser.add_argument("--use_confidence_mask",
                                   help="if set, replace rule-based auto-mask with learned confidence map",
                                   action="store_true")
          self.parser.add_argument("--confidence_reg_weight",
                                   type=float,
                                   help="regularization weight encouraging confidence to stay informative",
                                   default=0.01)
          self.parser.add_argument("--mypretrain",
                                   type=str,
                                   help="if set, use my pretrained encoder")
          self.parser.add_argument("--weights_init",
                                   type=str,
                                   help="pretrained or scratch",
                                   default="pretrained",
                                   choices=["pretrained", "scratch"])
          self.parser.add_argument("--pose_model_input",
                                   type=str,
                                   help="how many images the pose network gets",
                                   default="pairs",
                                   choices=["pairs", "all"])
          self.parser.add_argument("--pose_model_type",
                                   type=str,
                                   help="normal or shared",
                                   default="separate_resnet",
                                   choices=["posecnn", "separate_resnet", "shared"])

          # SYSTEM options
          self.parser.add_argument("--no_cuda",
                                   help="if set disables CUDA",
                                   action="store_true")
          self.parser.add_argument("--num_workers",
                                   type=int,
                                   help="number of dataloader workers",
                                   default=12)

          # LOADING options
          self.parser.add_argument("--load_weights_folder",
                                   type=str,
                                   help="name of model to load")
          self.parser.add_argument("--models_to_load",
                                   nargs="+",
                                   type=str,
                                   help="models to load",
                                   default=["encoder", "depth", "pose_encoder", "pose"])

          # LOGGING options
          self.parser.add_argument("--log_frequency",
                                   type=int,
                                   help="number of batches between each tensorboard log",
                                   default=250)
          self.parser.add_argument("--save_frequency",
                                   type=int,
                                   help="number of epochs between each save",
                                   default=1)

          # EVALUATION options
          self.parser.add_argument("--disable_median_scaling",
                                   help="if set disables median scaling in evaluation",
                                   action="store_true")
          self.parser.add_argument("--pred_depth_scale_factor",
                                   help="if set multiplies predictions by this number",
                                   type=float,
                                   default=1)
          self.parser.add_argument("--ext_disp_to_eval",
                                   type=str,
                                   help="optional path to a .npy disparities file to evaluate")
          self.parser.add_argument("--eval_split",
                                   type=str,
                                   default="eigen",
                                   choices=[
                                        "eigen"],
                                   help="which split to run eval on")
          self.parser.add_argument("--save_pred_disps",
                                   help="if set saves predicted disparities",
                                   action="store_true")
          self.parser.add_argument("--no_eval",
                                   help="if set disables evaluation",
                                   action="store_true")
          self.parser.add_argument("--eval_out_dir",
                                   help="if set will output the disparities to this folder",
                                   type=str)
          self.parser.add_argument("--post_process",
                                   help="if set will perform the flipping post processing "
                                        "from the original monodepth paper",
                                   action="store_true")
          self.parser.add_argument("--report_region_metrics",
                                   help="if set, report boundary/non-boundary/contour-vicinity metrics",
                                   action="store_true")
          self.parser.add_argument("--boundary_band_px",
                                   type=int,
                                   help="pixel band radius for GT depth edge boundary region",
                                   default=1)
          self.parser.add_argument("--contour_band_px",
                                   type=int,
                                   help="pixel band radius for GT contour vicinity region",
                                   default=2)
          self.parser.add_argument("--eval_confidence",
                                   help="if set, load and run confidence head during evaluation",
                                   action="store_true")
          self.parser.add_argument("--save_confidence_vis",
                              help="if set, save confidence visualization images",
                              action="store_true")
          self.parser.add_argument("--confidence_vis_dir",
                                   type=str,
                                   help="directory to save confidence visualization images",
                                   default="")
          self.parser.add_argument("--confidence_vis_count",
                                   type=int,
                                   help="number of samples to save confidence visualization for",
                                   default=20)
          self.parser.add_argument("--report_confidence_stats",
                                   help="if set, report confidence statistics on full/boundary/non-boundary regions",
                                   action="store_true")
          self.parser.add_argument("--use_geometric_loss",
                                   help="if set, add geometric consistency loss on top of photometric loss",
                                   action="store_true")
          self.parser.add_argument("--geometric_loss_only",
                                   help="if set, use ONLY geometric consistency loss (no photometric loss)",
                                   action="store_true")
          self.parser.add_argument("--geometric_loss_weight",
                                   type=float,
                                   help="weight for geometric consistency loss when combined with photometric",
                                   default=1.0)

          # =====================================================================
          # Sampling Strategy Options (深度与姿态联合采样策略)
          # =====================================================================
          self.parser.add_argument("--use_sampling_strategy",
                                   help="if set, enable depth and pose sampling strategy",
                                   action="store_true")
          self.parser.add_argument("--depth_num_samples",
                                   type=int,
                                   help="number of depth samples for sampling strategy",
                                   default=8)
          self.parser.add_argument("--depth_sampling_mode",
                                   type=str,
                                   help="depth sampling space: scaled_disp (relative perturbation recommended)",
                                   default="scaled_disp",
                                   choices=["disp", "scaled_disp", "log_depth"])
          self.parser.add_argument("--depth_sampling_noise",
                                   type=float,
                                   help="depth sampling noise std (for log/relative space)",
                                   default=0.03)
          self.parser.add_argument("--pose_num_samples",
                                   type=int,
                                   help="number of pose samples for sampling strategy",
                                   default=8)
          self.parser.add_argument("--pose_rot_noise",
                                   type=float,
                                   help="rotation sampling noise in radians (rad)",
                                   default=0.005)
          self.parser.add_argument("--pose_trans_noise_ratio",
                                   type=float,
                                   help="translation relative perturbation ratio",
                                   default=0.03)
          self.parser.add_argument("--depth_consistency_weight",
                                   type=float,
                                   help="weight for depth consistency constraint in sampling",
                                   default=0.01)
          self.parser.add_argument("--pose_consistency_weight",
                                   type=float,
                                   help="weight for pose consistency constraint in sampling",
                                   default=0.002)
          self.parser.add_argument("--sampling_margin_rel",
                                   type=float,
                                   help="relative margin for adopting a noisy (depth, pose) "
                                        "candidate over the clean-clean baseline; a noisy winner "
                                        "must be lower than clean * (1 - margin_rel)",
                                   default=0.05)
          self.parser.add_argument("--sampling_margin_abs",
                                   type=float,
                                   help="absolute margin threshold for adopting a noisy candidate; "
                                        "effective threshold = clean - max(margin_abs, margin_rel * clean)",
                                   default=0.0)
          self.parser.add_argument("--sampling_distill_d_weight",
                                   type=float,
                                   help="weight for depth distillation ||disp_hat - stopgrad(D_best)||; "
                                        "sampling is auxiliary only, keep this small",
                                   default=0.01)
          self.parser.add_argument("--sampling_distill_t_weight",
                                   type=float,
                                   help="weight for pose distillation ||T_hat - stopgrad(T_best)||; "
                                        "sampling is auxiliary only, keep this small",
                                   default=0.01)

          self.parser.add_argument("--config",
                                   type=str,
                                   default=None,
                                   help="path to YAML config file (all args can be set via YAML)")

     def parse(self):
          self.options = self.parser.parse_args()
          return self.options
