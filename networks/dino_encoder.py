import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
torch.hub.set_dir("/data3/jzy/.cache/torch/hub")


class ConvRefine(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DINOEncoder(nn.Module):
    """
    DINOv2 encoder adapter for Lite-Mono / Monodepth-style decoders.

    Uses official facebookresearch/dinov2 torch.hub backbone.
    Returns 3 feature maps at roughly 1/4, 1/8, 1/16 resolutions.
    """

    def __init__(
        self,
        model_name="dinov2_vits14",
        out_channels=(48, 80, 128),
        out_scales=(4, 8, 16),
        freeze_backbone=True,
        unfreeze_last_n_blocks=0,
    ):
        super().__init__()

        self.model_name = model_name
        self.out_scales = out_scales
        self.patch_size = 14

        # ---- official DINOv2 backbone ----
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)

        # infer embed dim robustly
        embed_dim = None
        if hasattr(self.backbone, "embed_dim"):
            embed_dim = self.backbone.embed_dim
        elif hasattr(self.backbone, "blocks") and len(self.backbone.blocks) > 0:
            # works for standard ViT block structure
            if hasattr(self.backbone.blocks[0].attn, "qkv"):
                embed_dim = self.backbone.blocks[0].attn.qkv.in_features

        if embed_dim is None:
            raise ValueError(f"Unable to infer embed_dim from model {model_name}")

        self.embed_dim = embed_dim
        self.num_ch_enc = np.array(list(out_channels))

        # 3 projected pyramid levels for your depth decoder
        self.proj_4 = ConvRefine(embed_dim, out_channels[0])
        self.proj_8 = ConvRefine(embed_dim, out_channels[1])
        self.proj_16 = ConvRefine(embed_dim, out_channels[2])

        # by default: freeze backbone, only train adapter + decoder
        if freeze_backbone:
            self.freeze_backbone()
        else:
            self.unfreeze_backbone()

        # optional partial unfreeze
        if unfreeze_last_n_blocks > 0:
            self.unfreeze_last_n_blocks(unfreeze_last_n_blocks)

    # ---------------------------
    # freeze / unfreeze controls
    # ---------------------------
    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def unfreeze_last_n_blocks(self, n=2):
        """
        Freeze all backbone params, then unfreeze only the last n transformer blocks
        plus final norm / head-norm-like layers if present.
        """
        self.freeze_backbone()

        if not hasattr(self.backbone, "blocks"):
            raise ValueError("Backbone does not expose .blocks, cannot partially unfreeze.")

        n = min(n, len(self.backbone.blocks))
        for blk in self.backbone.blocks[-n:]:
            for p in blk.parameters():
                p.requires_grad = True

        # final norm
        if hasattr(self.backbone, "norm") and self.backbone.norm is not None:
            for p in self.backbone.norm.parameters():
                p.requires_grad = True

    def trainable_parameter_groups(self, backbone_lr=None, adapter_lr=None):
        """
        Optional helper if you want different lr for backbone and adapter.
        Example:
            groups = encoder.trainable_parameter_groups(backbone_lr=1e-5, adapter_lr=1e-4)
        """
        backbone_params = []
        adapter_params = []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("backbone."):
                backbone_params.append(p)
            else:
                adapter_params.append(p)

        groups = []
        if backbone_params:
            groups.append({
                "params": backbone_params,
                "lr": backbone_lr if backbone_lr is not None else 1e-5
            })
        if adapter_params:
            groups.append({
                "params": adapter_params,
                "lr": adapter_lr if adapter_lr is not None else 1e-4
            })
        return groups

    # ---------------------------
    # feature utilities
    # ---------------------------
    def _tokens_to_map(self, tokens, h, w):
        """
        tokens: [B, N, C]
        return: [B, C, H/14, W/14]
        """
        ph = h // self.patch_size
        pw = w // self.patch_size
        expected = ph * pw

        if tokens.shape[1] != expected:
            raise ValueError(
                f"Token count mismatch: got {tokens.shape[1]}, expected {expected} "
                f"for input {(h, w)} with patch size {self.patch_size}"
            )

        feat = tokens.permute(0, 2, 1).reshape(tokens.shape[0], tokens.shape[-1], ph, pw)
        return feat

    def _resize_to_scale(self, feat, h, w, scale):
        return F.interpolate(
            feat,
            size=(max(1, h // scale), max(1, w // scale)),
            mode="bilinear",
            align_corners=False,
        )

    # ---------------------------
    # forward
    # ---------------------------
    def forward(self, x):
        """
        x: [B, 3, H, W]
        H and W must be divisible by 14
        """
        b, _, h, w = x.shape

        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(
                f"Input size {(h, w)} must be divisible by patch size {self.patch_size}"
            )

        # get 4 intermediate layers from official DINOv2 backbone
        # returns list of tensors [B, N, C]
        feats = self.backbone.get_intermediate_layers(
            x,
            n=4,
            reshape=False,
            return_class_token=False,
            norm=True,
        )

        if not isinstance(feats, (list, tuple)) or len(feats) < 3:
            raise ValueError(f"Expected >=3 intermediate layers, got {type(feats)} len={len(feats) if isinstance(feats, (list, tuple)) else 'NA'}")

        # choose shallow / middle / deep
        t4 = feats[0]
        t8 = feats[1]
        t16 = feats[-1]

        m4 = self._tokens_to_map(t4, h, w)
        m8 = self._tokens_to_map(t8, h, w)
        m16 = self._tokens_to_map(t16, h, w)

        f4 = self._resize_to_scale(m4, h, w, self.out_scales[0])
        f8 = self._resize_to_scale(m8, h, w, self.out_scales[1])
        f16 = self._resize_to_scale(m16, h, w, self.out_scales[2])

        f4 = self.proj_4(f4)
        f8 = self.proj_8(f8)
        f16 = self.proj_16(f16)

        return [f4, f8, f16]