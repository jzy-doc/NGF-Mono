from __future__ import absolute_import, division, print_function


import time
import os
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks


# torch.backends.cudnn.benchmark = True


def expand_bchw(idx, ref_shape):
    """Expand index tensor to match reference shape for gather operation.
    
    Args:
        idx: [B, H, W] or [1, B, H, W] indices
        ref_shape: [N, B, C, H, W] target shape
    
    Returns:
        expanded indices with shape [N, B, C, H, W]
    """
    n, b, c, h, w = ref_shape
    if idx.dim() == 3:  # [B, H, W]
        idx = idx.unsqueeze(0)  # [1, B, H, W]
    # Expand: [1, B, H, W] -> [N, B, 1, H, W] -> [N, B, C, H, W]
    idx = idx.expand(n, b, 1, h, w)
    return idx


def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


class Trainer:
    def __init__(self, options, gpu_id=0, world_size=1):
        self.opt = options
        self.gpu_id = gpu_id
        self.world_size = world_size
        self.is_main = (gpu_id == 0)
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.models_pose = {}
        self.parameters_to_train = []
        self.parameters_to_train_pose = []

        if self.world_size > 1:
            # DDP: each process has its own GPU
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        self.profile = self.opt.profile

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])
        assert not (self.opt.use_edge_input and self.opt.use_edge_mid_fusion), \
            "use_edge_input and use_edge_mid_fusion are mutually exclusive"
        assert not (self.opt.use_edge_input and self.opt.use_neuro_fusion), \
            "use_edge_input and use_neuro_fusion are mutually exclusive"

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        if self.opt.depth_encoder_type == "dino":
            assert not self.opt.use_edge_input and not self.opt.use_edge_mid_fusion, \
                "Minimal DINO replacement keeps edge-input/fusion disabled."
            self.models["encoder"] = networks.DINOEncoder(
                model_name=self.opt.dino_model_name,
                # pretrained=self.opt.dino_pretrained,
                freeze_backbone=self.opt.freeze_dino_backbone)
            if self.opt.freeze_dino_backbone:
                print("DINO backbone is frozen (training adapter + decoder only).")
        else:
            encoder_in_chans = 4 if self.opt.use_edge_input else 3
            self.models["encoder"] = networks.LiteMono(
                model=self.opt.model,
                drop_path_rate=self.opt.drop_path,
                width=self.opt.width,
                height=self.opt.height,
                in_chans=encoder_in_chans,
                use_edge_mid_fusion=self.opt.use_edge_mid_fusion,
                edge_fusion_strength=self.opt.edge_fusion_strength,
                edge_fusion_strength_mode=self.opt.edge_fusion_strength_mode,
                edge_fusion_mode=self.opt.edge_fusion_mode,
                edge_stage_mode=self.opt.edge_stage_mode,
                # Neuro-inspired fusion options
                use_neuro_fusion=self.opt.use_neuro_fusion,
                neuro_fusion_mode=self.opt.neuro_fusion_mode,
                neuro_fusion_stage=self.opt.neuro_fusion_stage,
                neuro_multiscale=self.opt.neuro_multiscale)

        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc,
                                                     self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models_pose["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models_pose["pose_encoder"].to(self.device)
                self.parameters_to_train_pose += list(self.models_pose["pose_encoder"].parameters())

                self.models_pose["pose"] = networks.PoseDecoder(
                    self.models_pose["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models_pose["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models_pose["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models_pose["pose"].to(self.device)
            self.parameters_to_train_pose += list(self.models_pose["pose"].parameters())

        if self.opt.predictive_mask:
            assert not self.opt.use_confidence_mask, \
                "predictive_mask and use_confidence_mask cannot be enabled together"
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        if self.opt.use_confidence_mask:
            if not self.opt.disable_automasking:
                print("use_confidence_mask enabled: disable_automasking is forced to True")
                self.opt.disable_automasking = True
            self.models["confidence"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["confidence"].to(self.device)
            self.parameters_to_train += list(self.models["confidence"].parameters())

        # _unwrap helper (defined early so load_model / load_pretrain can use it)
        def _unwrap(model_or_ddp):
            if isinstance(model_or_ddp, DDP):
                return model_or_ddp.module
            return model_or_ddp
        self._unwrap = _unwrap
        self.encoder = _unwrap(self.models["encoder"])
        self.depth_decoder = _unwrap(self.models["depth"])

        self.sobel_kernel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0],
              [-2.0, 0.0, 2.0],
              [-1.0, 0.0, 1.0]]], device=self.device).unsqueeze(0)
        self.sobel_kernel_y = torch.tensor(
            [[[-1.0, -2.0, -1.0],
              [0.0, 0.0, 0.0],
              [1.0, 2.0, 1.0]]], device=self.device).unsqueeze(0)
        self.avg_kernel = torch.ones(1, 1, 1, 1, device=self.device)

        self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.lr[0], weight_decay=self.opt.weight_decay)
        if self.use_pose_net:
            self.model_pose_optimizer = optim.AdamW(self.parameters_to_train_pose, self.opt.lr[3], weight_decay=self.opt.weight_decay)

        self.model_lr_scheduler = ChainedScheduler(
                            self.model_optimizer,
                            T_0=int(self.opt.lr[2]),
                            T_mul=1,
                            eta_min=self.opt.lr[1],
                            last_epoch=-1,
                            max_lr=self.opt.lr[0],
                            warmup_steps=0,
                            gamma=0.9
                        )
        self.model_pose_lr_scheduler = ChainedScheduler(
            self.model_pose_optimizer,
            T_0=int(self.opt.lr[5]),
            T_mul=1,
            eta_min=self.opt.lr[4],
            last_epoch=-1,
            max_lr=self.opt.lr[3],
            warmup_steps=0,
            gamma=0.9
        )

        if self.opt.load_weights_folder is not None:
            self.load_model()
            # Update scheduler last_epoch to match loaded optimizer state
            self.model_lr_scheduler.last_epoch = self.opt.start_epoch
            self.model_pose_lr_scheduler.last_epoch = self.opt.start_epoch
            print(f"Resuming from epoch {self.opt.start_epoch}, learning rate restored")

        if self.opt.mypretrain is not None:
            self.load_pretrain()

        # ------------------------------------------------------------------
        # DDP: wrap models after all weights loaded so broadcast is correct
        # ------------------------------------------------------------------
        if self.world_size > 1:
            # Only wrap models that actually participate in the loss computation.
            # pose models only wrapped when use_pose_net is True.
            active_models = dict(self.models)
            if self.use_pose_net:
                active_models.update(self.models_pose)

            for name, model in active_models.items():
                self.models[name] = DDP(
                    model,
                    device_ids=[self.gpu_id],
                    output_device=self.gpu_id,
                    find_unused_parameters=True,
                    broadcast_buffers=False,
                )
            print(f"[GPU {self.gpu_id}] DDP initialized (world_size={self.world_size})")

            # CRITICAL: optimizer must reference DDP-wrapped parameters, not unwrapped.
            # After wrapping, re-collect the trainable parameters from the wrapped models.
            self.parameters_to_train = []
            self.parameters_to_train_pose = []
            for name, model in self.models.items():
                self.parameters_to_train += list(model.parameters())
            for name, model in self.models_pose.items():
                self.parameters_to_train_pose += list(model.parameters())

            # Rebuild optimizer with wrapped-model parameters
            self.model_optimizer = optim.AdamW(
                self.parameters_to_train, self.opt.lr[0], weight_decay=self.opt.weight_decay)
            if self.use_pose_net:
                self.model_pose_optimizer = optim.AdamW(
                    self.parameters_to_train_pose, self.opt.lr[3], weight_decay=self.opt.weight_decay)

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.log_path)
        print("Training is using:\n  ", self.device)

        # ------------------------------------------------------------------
        # DDP: wrap models after all weights loaded so broadcast is correct
        # ------------------------------------------------------------------
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        num_workers = self.opt.num_workers
        if self.opt.use_geometric_loss or self.opt.geometric_loss_only:
            num_workers = 0  # geometric loss doubles activation memory, disable multiprocessing

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)

        if self.world_size > 1:
            train_sampler = DistributedSampler(
                train_dataset, num_replicas=self.world_size,
                rank=self.gpu_id, shuffle=True, drop_last=True)
        else:
            train_sampler = None

        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size,
            shuffle=(train_sampler is None),  # shuffle OR sampler, not both
            num_workers=num_workers, pin_memory=True,
            drop_last=True, sampler=train_sampler)

        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        # Val loader: no sampler (only rank 0 uses it)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)  # initialised here, recreated per epoch in val()

        # Only rank 0 writes tensorboard
        self.writers = {}
        if self.is_main:
            for mode in ["train", "val"]:
                self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        if self.is_main:
            self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline."""
        self.epoch = self.opt.start_epoch
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.start_epoch, self.opt.num_epochs):
            self.run_epoch()
            # Only main rank saves checkpoints
            if self.is_main and (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()
            # Sync all ranks at epoch boundary
            if self.world_size > 1:
                dist.barrier()

    def run_epoch(self):
        """Run a single epoch of training and validation."""
        if self.world_size > 1:
            # Tell sampler which epoch this is (for reproducible shuffling)
            self.train_loader.sampler.set_epoch(self.epoch)

        if self.is_main:
            print(f"[Epoch {self.epoch}] Training")
        self.set_train()

        self.model_lr_scheduler.step()
        if self.use_pose_net:
            self.model_pose_lr_scheduler.step()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            if self.use_pose_net:
                self.model_pose_optimizer.zero_grad()
            
            # 检查 loss 是否有效
            if torch.isnan(losses["loss"]) or torch.isinf(losses["loss"]):
                print(f"[WARNING] Skipping batch {batch_idx}: loss is {losses['loss']}")
                continue
            
            losses["loss"].backward()
            
            # 梯度裁剪 - 防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.parameters_to_train, max_norm=1.0)
            if self.use_pose_net:
                torch.nn.utils.clip_grad_norm_(self.parameters_to_train_pose, max_norm=1.0)
            
            self.model_optimizer.step()
            if self.use_pose_net:
                self.model_pose_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 20000
            late_phase = self.step % 2000 == 0

            if (early_phase or late_phase) and self.is_main:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat(
                [self._prepare_encoder_input(inputs[("color_aug", i, 0)]) for i in self.opt.frame_ids]
            )
            if self.opt.use_edge_mid_fusion:
                all_edge = torch.cat(
                    [self._build_edge_for_midfusion(inputs[("color_aug", i, 0)]) for i in self.opt.frame_ids]
                )
                all_features = self.models["encoder"](all_color_aug, edge=all_edge)
            elif self.opt.use_neuro_fusion:
                # Neuro fusion - direct pass to encoder
                all_features = self.models["encoder"](all_color_aug)
            else:
                all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder

            if self.opt.use_edge_mid_fusion:
                edge = self._build_edge_for_midfusion(inputs["color_aug", 0, 0])
                features = self.models["encoder"](inputs["color_aug", 0, 0], edge=edge)
            elif self.opt.use_neuro_fusion:
                # Neuro fusion - direct pass to encoder (encoder does its own normalization)
                features = self.models["encoder"](inputs["color_aug", 0, 0])
            else:
                features = self.models["encoder"](self._prepare_encoder_input(inputs["color_aug", 0, 0]))

            outputs = self.models["depth"](features)

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)
        if self.opt.use_confidence_mask:
            outputs["confidence"] = self.models["confidence"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        # Predict depth for source frames when geometric loss is used
        if self.opt.use_geometric_loss or self.opt.geometric_loss_only:
            if self.opt.pose_model_type == "shared":
                for f_i in self.opt.frame_ids[1:]:
                    if f_i != "s":
                        feat = features[f_i]
                        disp_src = self.models["depth"](feat)
                        if self.opt.v1_multiscale:
                            for k, v in disp_src.items():
                                outputs[("disp", f_i, k)] = v
                        else:
                            for k, v in disp_src.items():
                                outputs[("disp", f_i)] = v
            else:
                for f_i in self.opt.frame_ids[1:]:
                    if f_i != "s":
                        edge_f = None
                        if self.opt.use_edge_mid_fusion:
                            edge_f = self._build_edge_for_midfusion(inputs["color_aug", f_i, 0])
                            feat_f = self.models["encoder"](inputs["color_aug", f_i, 0], edge=edge_f)
                        else:
                            feat_f = self.models["encoder"](
                                self._prepare_encoder_input(inputs["color_aug", f_i, 0]))
                        disp_src = self.models["depth"](feat_f)
                        if self.opt.v1_multiscale:
                            for k, v in disp_src.items():
                                outputs[("disp", f_i, k)] = v
                        else:
                            for k, v in disp_src.items():
                                outputs[("disp", f_i)] = v

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def _build_edge_for_midfusion(self, rgb):
        """Build edge input for mid-fusion. Can be sobel or gaussian noise."""
        if getattr(self.opt, 'edge_midfusion_gaussian_noise', False):
            gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
            # Match sobel edge normalization
            return torch.clamp(
                torch.randn_like(gray) * 0.2 + 0.5, 0.0, 1.0)
        else:
            return self._build_sobel_edge(rgb)

    def _build_sobel_edge(self, rgb):
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        grad_x = F.conv2d(gray, self.sobel_kernel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_kernel_y, padding=1)
        mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + self.opt.edge_epsilon)
        max_val = mag.amax(dim=(2, 3), keepdim=True)
        return mag / (max_val + self.opt.edge_epsilon)

    def _build_fourth_channel(self, rgb):
        mode = self.opt.fourth_channel_mode
        edge = self._build_sobel_edge(rgb)
        if mode == "edge":
            return edge
        if mode == "zero":
            return torch.zeros_like(edge)
        if mode == "gaussian":
            return torch.clamp(
                torch.randn_like(edge) * self.opt.fourth_channel_noise_std + 0.5, 0.0, 1.0)
        if mode == "shuffled_edge":
            b, _, h, w = edge.shape
            flat = edge.reshape(b, 1, -1)
            perm = torch.argsort(torch.rand(b, 1, h * w, device=edge.device), dim=2)
            shuffled = torch.gather(flat, 2, perm).reshape(b, 1, h, w)
            return shuffled
        if mode == "blurred_edge":
            k = max(1, int(self.opt.fourth_channel_blur_kernel))
            if k % 2 == 0:
                k += 1
            return F.avg_pool2d(edge, kernel_size=k, stride=1, padding=k // 2)
        raise ValueError("Unsupported fourth_channel_mode: {}".format(mode))

    def _prepare_encoder_input(self, color_aug):
        if not self.opt.use_edge_input:
            return color_aug
        fourth = self._build_fourth_channel(color_aug)
        return torch.cat([color_aug, fourth], dim=1)

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models_pose["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models_pose["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models_pose["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        # try:
        #     inputs = self.val_iter.next()
        # except StopIteration:
        #     self.val_iter = iter(self.val_loader)
        #     inputs = self.val_iter.next()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_geometric_loss(self, inputs, outputs):
        """Compute geometric consistency loss: the predicted depth should be consistent
        across views after applying the predicted relative pose.

        For each pair (frame_0, frame_id):
        - Forward: project frame_0 pixel with depth D_0 to frame_id, compare with D_id at projected location.
        - Backward: project frame_id pixel with depth D_id to frame_0, compare with D_0 at projected location.

        Loss is computed at the native resolution of disp output (1/4 of full resolution).
        """
        total_geo_loss = 0
        batch_size = self.opt.batch_size

        K = inputs[("K", 0)]
        inv_K = inputs[("inv_K", 0)]

        disp_0 = outputs[("disp", 0)]
        _, depth_0 = disp_to_depth(disp_0, self.opt.min_depth, self.opt.max_depth)
        h_d, w_d = disp_0.shape[2], disp_0.shape[3]
        h_full = h_d
        w_full = w_d
        K_s = K
        inv_K_s = inv_K

        meshgrid = np.meshgrid(range(w_d), range(h_d), indexing='xy')
        id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        id_coords = torch.from_numpy(id_coords).to(self.device)
        ones = torch.ones(batch_size, 1, h_d * w_d, device=self.device)
        pix_coords = torch.unsqueeze(
            torch.stack([id_coords[0].view(-1), id_coords[1].view(-1)], 0), 0)
        pix_coords = pix_coords.repeat(batch_size, 1, 1).to(self.device)
        pix_coords_cat = torch.cat([pix_coords, ones], 1)

        cam_points_0 = torch.matmul(inv_K_s[:, :3, :3], pix_coords_cat)
        cam_points_0 = depth_0.view(batch_size, 1, -1) * cam_points_0
        cam_points_0_h = torch.cat([cam_points_0, ones], 1)

        for frame_id in self.opt.frame_ids[1:]:
            if frame_id == "s":
                continue

            T_0_to_id = outputs[("cam_T_cam", 0, frame_id)]

            if self.opt.pose_model_type == "posecnn":
                axisangle = outputs[("axisangle", 0, frame_id)]
                translation = outputs[("translation", 0, frame_id)]
                inv_depth = 1 / depth_0
                mean_inv_depth = inv_depth.mean(3, True).mean(2, True)
                T_0_to_id = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

            disp_id = outputs[("disp", frame_id)]
            _, depth_id = disp_to_depth(disp_id, self.opt.min_depth, self.opt.max_depth)

            P_0_to_id = torch.matmul(K_s, T_0_to_id)[:, :3, :]
            cam_points_id = torch.matmul(P_0_to_id, cam_points_0_h)
            depth_transformed_id = cam_points_id[:, 2:3, :]

            pix_coords_0_to_id = cam_points_id[:, :2, :] / (
                cam_points_id[:, 2:3, :] + 1e-7)
            pix_coords_0_to_id = pix_coords_0_to_id.view(
                batch_size, 2, h_d, w_d).permute(0, 2, 3, 1)
            pix_coords_0_to_id[..., 0] /= (w_d - 1)
            pix_coords_0_to_id[..., 1] /= (h_d - 1)
            pix_coords_0_to_id = (pix_coords_0_to_id - 0.5) * 2

            depth_id_sampled = F.grid_sample(
                depth_id, pix_coords_0_to_id,
                padding_mode="border", align_corners=True)
            depth_id_sampled = depth_id_sampled.view(batch_size, 1, h_d, w_d)

            geo_loss_forward = torch.abs(
                depth_transformed_id.view(batch_size, 1, h_d, w_d) - depth_id_sampled)
            total_geo_loss += geo_loss_forward.mean()

            inv_T_0_to_id = torch.inverse(T_0_to_id)
            P_id_to_0 = torch.matmul(K_s, inv_T_0_to_id)[:, :3, :]

            cam_points_id_src = torch.matmul(inv_K_s[:, :3, :3], pix_coords_cat)
            cam_points_id_src = depth_id.view(batch_size, 1, -1) * cam_points_id_src
            cam_points_id_src_h = torch.cat([cam_points_id_src, ones], 1)

            cam_points_0_back = torch.matmul(P_id_to_0, cam_points_id_src_h)
            depth_transformed_0 = cam_points_0_back[:, 2:3, :]
            pix_coords_id_to_0 = cam_points_0_back[:, :2, :] / (
                cam_points_0_back[:, 2:3, :] + 1e-7)
            pix_coords_id_to_0 = pix_coords_id_to_0.view(
                batch_size, 2, h_d, w_d).permute(0, 2, 3, 1)
            pix_coords_id_to_0[..., 0] /= (w_d - 1)
            pix_coords_id_to_0[..., 1] /= (h_d - 1)
            pix_coords_id_to_0 = (pix_coords_id_to_0 - 0.5) * 2

            depth_0_sampled = F.grid_sample(
                depth_0, pix_coords_id_to_0,
                padding_mode="border", align_corners=True)
            depth_0_sampled = depth_0_sampled.view(batch_size, 1, h_d, w_d)

            geo_loss_backward = torch.abs(
                depth_transformed_0.view(batch_size, 1, h_d, w_d) - depth_0_sampled)
            total_geo_loss += geo_loss_backward.mean()

        return total_geo_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """

        losses = {}
        total_loss = 0

        if self.opt.geometric_loss_only:
            geo_loss = self.compute_geometric_loss(inputs, outputs)
            losses["loss"] = geo_loss
            losses["geo_loss"] = geo_loss
            outputs["identity_selection/0"] = torch.zeros(
                self.opt.batch_size, self.opt.height, self.opt.width, device=self.device)
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids[1:]:
                    outputs.setdefault(("color", frame_id, s),
                        torch.zeros_like(inputs[("color", 0, s)]))
            return losses

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                color = inputs[("color", 0, scale)]
            else:
                color = inputs[("color", 0, 0)]
            target = inputs[("color", 0, source_scale)]

            # =============================================================
            # 始终计算标准重投影损失 (SSIM + L1 + automasking)
            # 这是主监督信号，与 use_sampling_strategy 无关。
            # 采样策略只在此之上额外加一个 detached search + 蒸馏辅助项。
            # =============================================================
            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            # 计算 identity reprojection losses for automasking
            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    self.compute_reprojection_loss(pred, target))
            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

            if self.opt.use_confidence_mask:
                confidence = outputs["confidence"][("disp", scale)]
                if not self.opt.v1_multiscale:
                    confidence = F.interpolate(
                        confidence, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)
                confidence = torch.clamp(confidence, min=1e-3, max=1.0)
                weighted_reprojection = reprojection_losses * confidence
                denom = confidence.sum(1, keepdim=True) + 1e-7
                conf_loss = weighted_reprojection.sum(1, keepdim=True) / denom

                # 结合 automasking: 与 identity loss 比较取最小
                if not self.opt.disable_automasking:
                    if self.opt.avg_reprojection:
                        identity_loss = identity_reprojection_losses.mean(1, keepdim=True)
                    else:
                        identity_loss = identity_reprojection_losses
                    combined = torch.cat((identity_loss, conf_loss), dim=1)
                    to_optimise, idxs = torch.min(combined, dim=1)
                    outputs["identity_selection/{}".format(scale)] = (
                        idxs > identity_loss.shape[1] - 1).float()
                else:
                    to_optimise = conf_loss

                loss += to_optimise.mean()

                confidence_reg = (1.0 - confidence).mean()
                loss += self.opt.confidence_reg_weight * confidence_reg
                losses["confidence_reg/{}".format(scale)] = confidence_reg

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if not self.opt.use_confidence_mask:
                if self.opt.avg_reprojection:
                    reprojection_loss = reprojection_losses.mean(1, keepdim=True)
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    reprojection_loss = reprojection_losses
                    identity_reprojection_loss = identity_reprojection_losses

                if not self.opt.disable_automasking:
                    # add random numbers to break ties
                    identity_reprojection_loss = identity_reprojection_loss + torch.randn(
                        identity_reprojection_loss.shape, device=self.device) * 0.00001

                    combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
                else:
                    combined = reprojection_loss

                if combined.shape[1] == 1:
                    to_optimise = combined
                else:
                    to_optimise, idxs = torch.min(combined, dim=1)

                if not self.opt.disable_automasking:
                    outputs["identity_selection/{}".format(scale)] = (
                        idxs > identity_reprojection_loss.shape[1] - 1).float()

                loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            if self.opt.v1_multiscale:
                color_for_smooth = color
            else:
                color_for_smooth = F.interpolate(
                    color, size=(disp.shape[2], disp.shape[3]),
                    mode="bilinear", align_corners=False)
            smooth_loss = get_smooth_loss(norm_disp, color_for_smooth)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss

        # Geometric consistency loss (only at scale 0)
        if self.opt.use_geometric_loss:
            geo_loss = self.compute_geometric_loss(inputs, outputs)
            losses["loss"] = total_loss + self.opt.geometric_loss_weight * geo_loss
            losses["geo_loss"] = geo_loss

        # Sampling strategy loss: detached search + 蒸馏辅助项
        # 核心原则:
        #   - 标准 photometric loss 已经在上面算完 (line ~860)，是主监督
        #   - 采样只用来在 detached 候选池里搜索一个更好的连贯候选 D_best/T_best
        #   - 候选池: [clean disp_hat] + N 个噪声样本 (clean 在 index 0)
        #   - 按样本均值 (不是逐像素) argmin 选一个 coherent 候选
        #   - 用蒸馏项 ||D_pred - stopgrad(D_best)|| + pose 同理轻微牵引
        if self.opt.use_sampling_strategy:
            sampling_losses = self.compute_sampling_losses(inputs, outputs)
            losses["sampling_distill_d"] = sampling_losses["distill_d"]
            losses["sampling_distill_t"] = sampling_losses["distill_t"]
            losses["sampling_best_reproj"] = sampling_losses["reproj_best"]
            losses["sampling_identity"] = sampling_losses["identity_loss"]
            losses["sampling_was_better"] = sampling_losses["was_better_frac"]

            # 蒸馏权重 (以 baseline 主损失为主，蒸馏只做轻微牵引):
            #   --sampling_distill_d_weight (默认 0.05)
            #   --sampling_distill_t_weight (默认 0.05)
            distill_d_w = getattr(self.opt, 'sampling_distill_d_weight', 0.05)
            distill_t_w = getattr(self.opt, 'sampling_distill_t_weight', 0.05)
            losses["loss"] = (
                losses["loss"]
                + distill_d_w * sampling_losses["distill_d"]
                + distill_t_w * sampling_losses["distill_t"]
            )

        return losses

    def compute_sampling_losses(self, inputs, outputs):
        """
        Detached-search + margin-gated distillation sampling strategy.

        The standard Monodepth2 photometric loss (SSIM+L1+automasking) computed on
        disp_hat / pose_hat remains the primary optimisation signal, handled in
        compute_losses. Sampling adds a secondary "search + distill" signal only:

        1. Candidate pool per batch sample = [clean depth + clean pose] + N_d * N_p
           noisy (depth, pose) candidates. The whole search runs under
           torch.no_grad so it never builds a graph through the noise.

        2. Each candidate's reproj loss is reduced to a [B] scalar (mean over
           C, H, W). The clean-clean reproj is recorded separately as the
           STRICT baseline.

        3. Per batch sample, argmin over the noisy sub-table picks a candidate.
           It is adopted ONLY if it beats the clean-clean baseline by a margin
           (max(margin_abs, margin_rel * clean) — strictly lower). Otherwise
           the clean prediction is kept and the distill loss collapses to zero.

        4. Distillation: ||disp_hat - D_best.detach()||_1 (+ pose distill when
           pose sampling is on). Gradients flow only through disp_hat / pose_hat.

        5. Returned scalars (reproj_best, reproj_clean, identity_loss,
           was_better_frac) are monitoring only — they do NOT enter the loss.
        """
        margin_rel = float(getattr(self.opt, 'sampling_margin_rel', 0.05))   # 5%
        margin_abs = float(getattr(self.opt, 'sampling_margin_abs', 0.0))    # abs

        pose_distill_on = bool(
            self.opt.pose_num_samples > 0 and (
                self.opt.pose_rot_noise > 0 or self.opt.pose_trans_noise_ratio > 0)
        )

        total_distill_d     = 0.0
        total_distill_t     = 0.0
        total_reproj_best   = 0.0  # genuine winner mean reproj (monitor)
        total_reproj_clean  = 0.0  # clean-clean baseline (monitor)
        total_identity      = 0.0
        total_was_better    = 0.0

        for scale in self.opt.scales:
            disp_hat = outputs[("disp", scale)]          # [B,1,H,W]
            B = disp_hat.shape[0]

            source_scale = scale if self.opt.v1_multiscale else 0
            if not self.opt.v1_multiscale and scale > 0:
                disp_hat = F.interpolate(
                    disp_hat, [self.opt.height, self.opt.width],
                    mode="bilinear", align_corners=False)

            target = inputs[("color", 0, source_scale)]

            # ── Build candidate pool (all detached) under no_grad ────────────
            with torch.no_grad():
                # Depth candidates: index 0 = clean (detached)
                if self.opt.depth_num_samples > 0 and self.opt.depth_sampling_noise > 0:
                    noisy_disp = depth_sampling(
                        disp_hat.detach(),
                        self.opt.depth_num_samples,
                        self.opt.depth_sampling_noise,
                        mode=self.opt.depth_sampling_mode,
                        min_depth=self.opt.min_depth,
                        max_depth=self.opt.max_depth)
                    depth_cands = torch.cat(
                        [disp_hat.detach().unsqueeze(0), noisy_disp], dim=0)
                    n_d = depth_cands.shape[0]
                else:
                    depth_cands = disp_hat.detach().unsqueeze(0)
                    n_d = 1

                # Pose candidates per frame (index 0 = clean)
                if pose_distill_on:
                    pose_cands_per_frame = {}
                    for frame_id in self.opt.frame_ids[1:]:
                        if frame_id == "s":
                            continue
                        aa_hat = outputs[("axisangle", 0, frame_id)].detach()
                        tt_hat = outputs[("translation", 0, frame_id)].detach()
                        aa_noise, tt_noise = pose_sampling(
                            aa_hat, tt_hat,
                            self.opt.pose_num_samples,
                            self.opt.pose_rot_noise,
                            self.opt.pose_trans_noise_ratio)
                        aa_all = torch.cat([aa_hat.unsqueeze(0), aa_noise], dim=0)
                        tt_all = torch.cat([tt_hat.unsqueeze(0), tt_noise], dim=0)
                        pose_cands_per_frame[frame_id] = (aa_all, tt_all)
                    # n_p is consistent across frames (same sample count)
                    any_fid = next(iter(pose_cands_per_frame))
                    n_p = pose_cands_per_frame[any_fid][0].shape[0]
                else:
                    pose_cands_per_frame = {}
                    n_p = 1

                per_frame_records = []   # tuples: (frame_id, D_best, T_best_aa,
                                          #          T_best_tt, reproj_best_m,
                                          #          reproj_clean_m, identity_m,
                                          #          adopt_noisy_mask)

                for frame_id in self.opt.frame_ids[1:]:
                    if frame_id == "s":
                        continue

                    source = inputs[("color", frame_id, source_scale)]

                    # Identity loss as a scalar (no grad path)
                    identity_m = self.compute_reprojection_loss(
                        source, target).mean().detach()

                    aa_all_frame, tt_all_frame = (
                        pose_cands_per_frame[frame_id] if pose_distill_on
                        else (None, None))

                    # Reproj matrix [n_d, n_p, B]: each entry mean over (C,H,W)
                    reproj_matrix = torch.zeros(
                        n_d, n_p, B, device=disp_hat.device)
                    for di in range(n_d):
                        dc = depth_cands[di]                 # [B,1,H,W]
                        for pi in range(n_p):
                            if pose_distill_on:
                                aa = aa_all_frame[pi]
                                tt = tt_all_frame[pi]
                            else:
                                aa = outputs[("axisangle", 0, frame_id)].detach()
                                tt = outputs[("translation", 0, frame_id)].detach()
                            r = self._compute_reprojection_for_sampling(
                                inputs, outputs, dc, aa, tt, frame_id, scale)
                            reproj_matrix[di, pi] = r.mean(dim=[1, 2, 3])  # [B]

                    # Strict baseline: clean-clean pair at (di=0, pi=0)
                    reproj_clean = reproj_matrix[0, 0]              # [B]

                    # Noisy candidates: all (di, pi) except (0, 0)
                    mask = torch.ones(n_d, n_p, dtype=torch.bool,
                                      device=disp_hat.device)
                    mask[0, 0] = False
                    noisy_vals = reproj_matrix[mask]                # [N_noisy, B]
                    masked_idx_to_full = mask.nonzero(as_tuple=False)

                    # Best noisy candidate per batch sample
                    # Note: min(dim=0) returns torch.return_types.min with
                    # .values (the min values) and .indices (argmin).  Use the
                    # named accessors for clarity and version-robustness.
                    _min_res = noisy_vals.min(dim=0)
                    noisy_winner_rel = _min_res.indices.long()          # [B]
                    reproj_best_cand  = _min_res.values                  # [B]
                    del _min_res
                    b_idx = torch.arange(B, device=disp_hat.device).long()
                    # Construct the full (di, pi) coordinates for each winner
                    # using the masked-index mapping.
                    di_w = masked_idx_to_full[noisy_winner_rel, 0]  # [B]
                    pi_w = masked_idx_to_full[noisy_winner_rel, 1]  # [B]
                    # Sanity: re-fetch from the full reproj_matrix using the
                    # computed (di, pi) coords — guarantees consistency with
                    # the search.
                    reproj_best_cand = reproj_matrix[di_w, pi_w, b_idx]   # [B] 

                    # Margin gate: noisy winner must be strictly below
                    #   clean - max(margin_abs, margin_rel * clean)
                    threshold = reproj_clean - torch.clamp(
                        margin_abs + margin_rel * reproj_clean,
                        min=0.0)
                    adopt_noisy = reproj_best_cand < threshold      # [B] bool

                    # Default (do-not-adopt) = clean depth + clean pose
                    D_best = depth_cands[0]                         # [B,1,H,W]
                    if pose_distill_on:
                        T_best_aa = aa_all_frame[0]                 # [B,N,1,3]
                        T_best_tt = tt_all_frame[0]
                    else:
                        T_best_aa = outputs[("axisangle", 0, frame_id)].detach()
                        T_best_tt = outputs[("translation", 0, frame_id)].detach()

                    # Adopted versions for samples where adopt_noisy is True
                    if adopt_noisy.any():
                        D_best_winner = depth_cands[di_w, b_idx]
                        replace_d = adopt_noisy.view(B, 1, 1, 1)
                        D_best = torch.where(replace_d, D_best_winner, D_best)
                        if pose_distill_on:
                            aa_w = aa_all_frame[pi_w, b_idx]
                            tt_w = tt_all_frame[pi_w, b_idx]
                            replace_p = adopt_noisy.view(B, 1, 1, 1)
                            T_best_aa = torch.where(replace_p, aa_w, T_best_aa)
                            T_best_tt = torch.where(replace_p, tt_w, T_best_tt)

                    # Monitoring (before margin): "reproj_best" is the truly
                    # best noisy candidate where applicable; else falls back
                    # to clean so a low value here means search found nothing
                    # genuinely better.
                    reproj_best_m = torch.where(
                        adopt_noisy, reproj_best_cand, reproj_clean)

                    per_frame_records.append(
                        (frame_id, D_best, T_best_aa, T_best_tt,
                         reproj_best_m.mean(), reproj_clean.mean(),
                         identity_m.mean(), adopt_noisy))

            # ── Compute distillation losses OUTSIDE no_grad so gradients flow
            #    only through disp_hat / pose_hat.
            distill_d_per_frame = []
            distill_t_per_frame = []
            reproj_best_per_frame = []
            reproj_clean_per_frame = []
            identity_list_for_this_scale = []
            was_better_cnt = 0

            for (frame_id, D_best, T_best_aa, T_best_tt,
                 reproj_best_m, reproj_clean_m, identity_m,
                 adopt_noisy) in per_frame_records:

                D_best = D_best.detach()
                T_best_aa = T_best_aa.detach()
                T_best_tt = T_best_tt.detach()

                distill_d = torch.abs(disp_hat - D_best).mean()
                distill_d_per_frame.append(distill_d)

                if pose_distill_on:
                    aa_hat = outputs[("axisangle", 0, frame_id)]
                    tt_hat = outputs[("translation", 0, frame_id)]
                    distill_t = (
                        torch.abs(T_best_aa - aa_hat).mean()
                        + torch.abs(T_best_tt - tt_hat).mean()
                    )
                else:
                    distill_t = torch.tensor(0.0, device=disp_hat.device)
                distill_t_per_frame.append(distill_t)

                reproj_best_per_frame.append(reproj_best_m)
                reproj_clean_per_frame.append(reproj_clean_m)
                identity_list_for_this_scale.append(identity_m)
                was_better_cnt += adopt_noisy.sum().item()

            n_frames = max(len(per_frame_records), 1)
            total_reproj_best  += sum(reproj_best_per_frame)  / n_frames
            total_reproj_clean += sum(reproj_clean_per_frame) / n_frames
            total_identity     += sum(identity_list_for_this_scale) / n_frames
            total_distill_d    += sum(distill_d_per_frame)   / n_frames
            total_distill_t    += sum(distill_t_per_frame)   / n_frames
            total_was_better   += was_better_cnt / (B * n_frames)

        # Average across scales
        n_scales = max(len(self.opt.scales), 1)
        total_reproj_best  /= n_scales
        total_reproj_clean /= n_scales
        total_identity     /= n_scales
        total_distill_d    /= n_scales
        total_distill_t    /= n_scales
        total_was_better   /= n_scales

        return {
            "distill_d":        total_distill_d,
            "distill_t":        total_distill_t,
            "reproj_best":      total_reproj_best,
            "reproj_clean":     total_reproj_clean,
            "identity_loss":    total_identity,
            "was_better_frac":  total_was_better,
        }

    def _compute_reprojection_for_sampling(self, inputs, outputs, disp_candidate,
                                          axisangle_candidate, translation_candidate,
                                          frame_id, scale):
        """
        Per-candidate reprojection loss used by compute_sampling_losses.

        Same warp path as the standard photometric loss (disp_to_depth ->
        backproject -> project_3d -> grid_sample -> SSIM+L1), but driven by an
        arbitrary (disp, axisangle, translation) triple so the search can score
        noisy candidates against the clean baseline.

        All inputs here are intended to be detached at the call site (the
        search runs under torch.no_grad), so this function does not need to
        worry about graph propagation.

        Args:
            disp_candidate:        [B, 1, H, W] candidate disparity
            axisangle_candidate:   [B, N, 1, 3] or [B, 1, 3] candidate rotation
            translation_candidate: [B, N, 1, 3] or [B, 1, 3] candidate translation
            frame_id: source frame id (sign indicates forward/backward)
            scale:   network scale index

        Returns:
            reproj_loss: [B, 1, H, W] per-pixel SSIM+L1 reprojection loss
        """
        if self.opt.v1_multiscale:
            source_scale = scale
        else:
            source_scale = 0

        # disp -> depth, normalize shape for backproject_depth
        _, depth_candidate = disp_to_depth(
            disp_candidate, self.opt.min_depth, self.opt.max_depth)
        if depth_candidate.dim() == 4:
            depth_candidate = depth_candidate.squeeze(1)

        K = inputs[("K", source_scale)].detach()
        inv_K = inputs[("inv_K", source_scale)].detach()
        target = inputs[("color", 0, source_scale)].detach()
        source = inputs[("color", frame_id, source_scale)].detach()

        # axisangle_candidate may be [B, N_frames, 1, 3] (full pose batch) or
        # [B, 1, 3] (single source). Normalize to [B, 1, 3] for
        # transformation_from_parameters.
        if axisangle_candidate.dim() == 4:
            axisangle_flat = axisangle_candidate[:, 0, 0, :].unsqueeze(1)
        else:
            axisangle_flat = axisangle_candidate
        if translation_candidate.dim() == 4:
            translation_flat = translation_candidate[:, 0, 0, :].unsqueeze(1)
        else:
            translation_flat = translation_candidate

        if self.opt.pose_model_type == "posecnn":
            inv_depth = 1.0 / (depth_candidate + 1e-7)
            mean_inv_depth = inv_depth.mean(3, True).mean(2, True)
            T = transformation_from_parameters(
                axisangle_flat,
                translation_flat * mean_inv_depth[:, 0],
                frame_id < 0)
        else:
            T = transformation_from_parameters(
                axisangle_flat,
                translation_flat,
                invert=(frame_id < 0))

        cam_points = self.backproject_depth[source_scale](
            depth_candidate, inv_K)
        pix_coords = self.project_3d[source_scale](
            cam_points, K, T)
        del cam_points

        pred = F.grid_sample(
            source, pix_coords,
            padding_mode="border", align_corners=True)

        # Use the same SSIM+L1 mix as the main photometric loss for consistency.
        reproj_loss = self.compute_reprojection_loss(pred, target)
        return reproj_loss

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | lr {:.6f} |lr_p {:.6f} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, self.model_optimizer.state_dict()['param_groups'][0]['lr'],
                                  self.model_pose_optimizer.state_dict()['param_groups'][0]['lr'],
                                  batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                # disp: ensure exactly (1, H, W) or (3, H, W) for CHW
                disp_full = outputs[("disp", s)]  # (B, 1, H, W)
                disp_j = disp_full[j].clone().float()  # (1, H, W)
                # ensure 3 dimensions for CHW format
                if disp_j.dim() == 2:
                    disp_j = disp_j.unsqueeze(0)
                elif disp_j.dim() == 4 and disp_j.shape[0] == 1:
                    disp_j = disp_j.squeeze(0)
                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    disp_j, self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)
                elif self.opt.use_confidence_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "confidence_{}_{}/{}".format(frame_id, s, j),
                            outputs["confidence"][("disp", s)][j, f_idx][None, ...],
                            self.step)
                elif not self.opt.disable_automasking and "identity_selection/{}".format(s) in outputs:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk (only from main rank in DDP)."""
        # In DDP, only save from main process
        if self.world_size > 1 and not self.is_main:
            return
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = self._unwrap(model).state_dict()
            if model_name == 'encoder':
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        for model_name, model in self.models_pose.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = self._unwrap(model).state_dict()
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam_pose"))
        if self.use_pose_net:
            torch.save(self.model_pose_optimizer.state_dict(), save_path)

    def load_pretrain(self):
        self.opt.mypretrain = os.path.expanduser(self.opt.mypretrain)
        path = self.opt.mypretrain
        encoder = self._unwrap(self.models["encoder"])
        model_dict = encoder.state_dict()
        raw = torch.load(path, map_location=self.device)
        # Support both raw state_dict and {'model': ...} wrapper
        pretrained_dict = raw if isinstance(raw, dict) and 'model' not in raw else raw.get('model', raw)
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
            if (k in model_dict and not k.startswith('norm') and v.shape == model_dict[k].shape)
        }
        model_dict.update(pretrained_dict)
        encoder.load_state_dict(model_dict)
        print('mypretrain loaded.')

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        folder_name = os.path.basename(self.opt.load_weights_folder)
        if folder_name.startswith("weights_"):
            try:
                loaded_epoch = int(folder_name.split("_")[1])
                if self.opt.start_epoch == 0:
                    self.opt.start_epoch = loaded_epoch
                    print(f"Detected starting epoch {loaded_epoch} from folder name")
            except (IndexError, ValueError):
                pass

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))

            if n in ['pose_encoder', 'pose']:
                model = self._unwrap(self.models_pose[n])
                model_dict = model.state_dict()
                pretrained_dict = torch.load(path, map_location=self.device)
                pretrained_dict = {
                    k: v for k, v in pretrained_dict.items()
                    if (k in model_dict and v.shape == model_dict[k].shape)
                }
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
            else:
                model = self._unwrap(self.models[n])
                model_dict = model.state_dict()
                pretrained_dict = torch.load(path, map_location=self.device)
                pretrained_dict = {
                    k: v for k, v in pretrained_dict.items()
                    if (k in model_dict and v.shape == model_dict[k].shape)
                }
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)

        # loading adam state (only from main rank)
        if self.world_size > 1 and not self.is_main:
            return
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        optimizer_pose_load_path = os.path.join(self.opt.load_weights_folder, "adam_pose.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path, map_location=self.device)
            optimizer_pose_dict = torch.load(optimizer_pose_load_path, map_location=self.device)
            self.model_optimizer.load_state_dict(optimizer_dict)
            self.model_pose_optimizer.load_state_dict(optimizer_pose_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

