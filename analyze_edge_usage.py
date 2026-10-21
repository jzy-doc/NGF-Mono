from __future__ import absolute_import, division, print_function

import argparse
import os
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import datasets
import networks
from layers import disp_to_depth
from utils import readlines

cv2.setNumThreads(0)

SPLITS_DIR = os.path.join(os.path.dirname(__file__), "splits")
MIN_DEPTH = 1e-3
MAX_DEPTH = 80


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def build_sobel_edge(rgb, eps=1e-6):
    gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    sobel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        device=rgb.device,
        dtype=rgb.dtype,
    ).unsqueeze(0)
    sobel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        device=rgb.device,
        dtype=rgb.dtype,
    ).unsqueeze(0)
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + eps)
    max_val = mag.amax(dim=(2, 3), keepdim=True)
    return mag / (max_val + eps)


def build_fourth_channel(color, mode="edge", edge_epsilon=1e-6, noise_std=0.2, blur_kernel=5):
    edge = build_sobel_edge(color, eps=edge_epsilon)
    if mode == "edge":
        return edge
    if mode == "zero":
        return torch.zeros_like(edge)
    if mode == "gaussian":
        return torch.clamp(torch.randn_like(edge) * noise_std + 0.5, 0.0, 1.0)
    if mode == "shuffled_edge":
        b, _, h, w = edge.shape
        flat = edge.reshape(b, 1, -1)
        perm = torch.argsort(torch.rand(b, 1, h * w, device=edge.device), dim=2)
        return torch.gather(flat, 2, perm).reshape(b, 1, h, w)
    if mode == "blurred_edge":
        k = max(1, int(blur_kernel))
        if k % 2 == 0:
            k += 1
        return F.avg_pool2d(edge, kernel_size=k, stride=1, padding=k // 2)
    raise ValueError("Unsupported fourth_channel_mode: {}".format(mode))


def prepare_encoder_input(color, use_edge_input, edge_epsilon, fourth_channel_mode="edge",
                          fourth_channel_noise_std=0.2, fourth_channel_blur_kernel=5):
    if not use_edge_input:
        return color
    fourth = build_fourth_channel(
        color,
        mode=fourth_channel_mode,
        edge_epsilon=edge_epsilon,
        noise_std=fourth_channel_noise_std,
        blur_kernel=fourth_channel_blur_kernel,
    )
    return torch.cat([color, fourth], dim=1)


def first_layer_edge_weight_ratio(encoder):
    # First conv in LiteMono stem
    w = encoder.downsample_layers[0][0].conv.weight.detach()
    in_ch = w.shape[1]
    if in_ch < 4:
        return None
    rgb_norms = [torch.norm(w[:, i, :, :]).item() for i in range(3)]
    edge_norm = torch.norm(w[:, 3, :, :]).item()
    ratio = edge_norm / (sum(rgb_norms) / len(rgb_norms) + 1e-12)
    return {
        "in_channels": in_ch,
        "edge_norm": edge_norm,
        "rgb_norms": rgb_norms,
        "edge_over_rgb_mean": ratio,
    }


def get_eval_mask(gt_depth):
    gt_height, gt_width = gt_depth.shape[:2]
    mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
    crop = np.array(
        [0.40810811 * gt_height, 0.99189189 * gt_height, 0.03594771 * gt_width, 0.96405229 * gt_width]
    ).astype(np.int32)
    crop_mask = np.zeros(mask.shape)
    crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
    return np.logical_and(mask, crop_mask)


def build_boundary_masks(gt_depth, eval_mask, edge_threshold=0.05, boundary_band=3):
    gx = cv2.Sobel(gt_depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gt_depth, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    boundary = grad > edge_threshold
    if boundary_band > 0:
        kernel = np.ones((boundary_band * 2 + 1, boundary_band * 2 + 1), np.uint8)
        boundary = cv2.dilate(boundary.astype(np.uint8), kernel, iterations=1).astype(bool)
    boundary = np.logical_and(boundary, eval_mask)
    non_boundary = np.logical_and(eval_mask, np.logical_not(boundary))
    return boundary, non_boundary


def run_depth_eval(pred_disps, gt_depths, edge_threshold=0.05, boundary_band=3):
    errors = []
    boundary_errors = []
    non_boundary_errors = []
    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]
        pred_disp = cv2.resize(pred_disps[i], (gt_width, gt_height))
        pred_depth = 1.0 / pred_disp

        mask = get_eval_mask(gt_depth)
        boundary_mask, non_boundary_mask = build_boundary_masks(
            gt_depth, mask, edge_threshold=edge_threshold, boundary_band=boundary_band)

        pred_depth_eval = pred_depth[mask]
        gt_eval = gt_depth[mask]

        # mono evaluation with median scaling
        ratio = np.median(gt_eval) / np.median(pred_depth_eval)
        pred_depth = np.clip(pred_depth * ratio, MIN_DEPTH, MAX_DEPTH)

        errors.append(compute_errors(gt_eval, pred_depth[mask]))
        if boundary_mask.sum() > 0:
            boundary_errors.append(compute_errors(gt_depth[boundary_mask], pred_depth[boundary_mask]))
        if non_boundary_mask.sum() > 0:
            non_boundary_errors.append(compute_errors(gt_depth[non_boundary_mask], pred_depth[non_boundary_mask]))

    out = {"global": np.array(errors).mean(0)}
    out["boundary"] = np.array(boundary_errors).mean(0) if boundary_errors else None
    out["non_boundary"] = np.array(non_boundary_errors).mean(0) if non_boundary_errors else None
    return out


def main():
    parser = argparse.ArgumentParser("Analyze edge channel usage for my-mono")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--load_weights_folder", type=str, required=True)
    parser.add_argument("--model", type=str, default="lite-mono",
                        choices=["lite-mono", "lite-mono-small", "lite-mono-tiny", "lite-mono-8m"])
    parser.add_argument("--eval_split", type=str, default="eigen")
    parser.add_argument("--png", action="store_true")
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=100.0)
    parser.add_argument("--use_edge_input", action="store_true")
    parser.add_argument("--edge_epsilon", type=float, default=1e-6)
    parser.add_argument("--primary_fourth_channel_mode", type=str, default="edge",
                        choices=["edge", "zero", "gaussian", "shuffled_edge", "blurred_edge"])
    parser.add_argument("--ablation_fourth_channel_mode", type=str, default="zero",
                        choices=["edge", "zero", "gaussian", "shuffled_edge", "blurred_edge"])
    parser.add_argument("--fourth_channel_noise_std", type=float, default=0.2)
    parser.add_argument("--fourth_channel_blur_kernel", type=int, default=5)
    parser.add_argument("--boundary_edge_threshold", type=float, default=0.05)
    parser.add_argument("--boundary_band", type=int, default=3)
    parser.add_argument("--max_batches", type=int, default=0,
                        help="0 means full set, >0 means only first N batches for quick diagnostics")
    args = parser.parse_args()

    load_weights_folder = os.path.expanduser(args.load_weights_folder)
    encoder_path = os.path.join(load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(load_weights_folder, "depth.pth")
    assert os.path.isdir(load_weights_folder), f"Cannot find folder {load_weights_folder}"

    filenames = readlines(os.path.join(SPLITS_DIR, args.eval_split, "test_files.txt"))
    encoder_dict = torch.load(encoder_path)
    decoder_dict = torch.load(decoder_path)

    img_ext = ".png" if args.png else ".jpg"
    dataset = datasets.KITTIRAWDataset(
        args.data_path,
        filenames,
        encoder_dict["height"],
        encoder_dict["width"],
        [0],
        4,
        is_train=False,
        img_ext=img_ext,
    )
    dataloader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = networks.LiteMono(
        model=args.model,
        height=encoder_dict["height"],
        width=encoder_dict["width"],
        in_chans=4 if args.use_edge_input else 3,
    ).to(device).eval()
    decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3)).to(device).eval()

    model_dict = encoder.state_dict()
    decoder_model_dict = decoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict and v.shape == model_dict[k].shape})
    decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in decoder_model_dict and v.shape == decoder_model_dict[k].shape})

    ratio_info = first_layer_edge_weight_ratio(encoder)
    print("=== 1) First-layer channel weight analysis ===")
    if ratio_info is None:
        print("Encoder first conv has <4 input channels. Run with --use_edge_input for edge model.")
    else:
        print("in_channels:", ratio_info["in_channels"])
        print("rgb_norms:", [round(x, 6) for x in ratio_info["rgb_norms"]])
        print("edge_norm:", round(ratio_info["edge_norm"], 6))
        print("edge_over_rgb_mean:", round(ratio_info["edge_over_rgb_mean"], 6))

    hook_modules = {
        "stem1_conv0": encoder.downsample_layers[0][0],
        "stem2_conv": encoder.stem2[0],
        "stage0_last": encoder.stages[0][-1],
        "stage1_last": encoder.stages[1][-1],
        "stage2_last": encoder.stages[2][-1],
    }
    feature_cache = {"primary": {}, "ablation": {}}
    mode = {"tag": "primary"}

    def make_hook(name):
        def _hook(_module, _inp, out):
            feature_cache[mode["tag"]][name] = out.detach()
        return _hook

    handles = [m.register_forward_hook(make_hook(n)) for n, m in hook_modules.items()]

    pred_disps_primary = []
    pred_disps_ablation = []
    feature_deltas = defaultdict(list)

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            color = data[("color", 0, 0)].to(device, non_blocking=True)

            mode["tag"] = "primary"
            input_primary = prepare_encoder_input(
                color, args.use_edge_input, args.edge_epsilon,
                fourth_channel_mode=args.primary_fourth_channel_mode,
                fourth_channel_noise_std=args.fourth_channel_noise_std,
                fourth_channel_blur_kernel=args.fourth_channel_blur_kernel)
            out_primary = decoder(encoder(input_primary))
            disp_primary, _ = disp_to_depth(out_primary[("disp", 0)], args.min_depth, args.max_depth)
            pred_disps_primary.append(disp_primary.cpu()[:, 0].numpy())

            mode["tag"] = "ablation"
            input_ablation = prepare_encoder_input(
                color, args.use_edge_input, args.edge_epsilon,
                fourth_channel_mode=args.ablation_fourth_channel_mode,
                fourth_channel_noise_std=args.fourth_channel_noise_std,
                fourth_channel_blur_kernel=args.fourth_channel_blur_kernel)
            out_ablation = decoder(encoder(input_ablation))
            disp_ablation, _ = disp_to_depth(out_ablation[("disp", 0)], args.min_depth, args.max_depth)
            pred_disps_ablation.append(disp_ablation.cpu()[:, 0].numpy())

            for name in hook_modules.keys():
                f1 = feature_cache["primary"].get(name)
                f2 = feature_cache["ablation"].get(name)
                if f1 is None or f2 is None:
                    continue
                abs_delta = torch.mean(torch.abs(f1 - f2)).item()
                base_mag = torch.mean(torch.abs(f1)).item() + 1e-12
                rel_delta = abs_delta / base_mag
                feature_deltas[name].append((abs_delta, rel_delta))

            if args.max_batches > 0 and (batch_idx + 1) >= args.max_batches:
                break

    for h in handles:
        h.remove()

    pred_disps_primary = np.concatenate(pred_disps_primary)
    pred_disps_ablation = np.concatenate(pred_disps_ablation)

    gt_path = os.path.join(SPLITS_DIR, args.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, allow_pickle=True, fix_imports=True, encoding="latin1")["data"]
    gt_depths = gt_depths[:pred_disps_primary.shape[0]]

    res_primary = run_depth_eval(
        pred_disps_primary, gt_depths,
        edge_threshold=args.boundary_edge_threshold, boundary_band=args.boundary_band)
    res_ablation = run_depth_eval(
        pred_disps_ablation, gt_depths,
        edge_threshold=args.boundary_edge_threshold, boundary_band=args.boundary_band)

    print("\n=== 2) Feature response difference (primary vs ablation) ===")
    for name, values in feature_deltas.items():
        mean_abs = float(np.mean([v[0] for v in values]))
        mean_rel = float(np.mean([v[1] for v in values]))
        print(f"{name:12s} | mean_abs_delta={mean_abs:.6e} | mean_rel_delta={mean_rel:.6e}")

    print("\n=== 3) Ablation performance comparison ===")
    metric_names = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
    print("primary mode :", args.primary_fourth_channel_mode)
    print("ablation mode:", args.ablation_fourth_channel_mode)
    print("global primary :", {k: round(v, 4) for k, v in zip(metric_names, res_primary["global"])})
    print("global ablation:", {k: round(v, 4) for k, v in zip(metric_names, res_ablation["global"])})
    print("global delta(ablation - primary):", {
        k: round(v2 - v1, 4) for k, v1, v2 in zip(metric_names, res_primary["global"], res_ablation["global"])
    })

    if res_primary["boundary"] is not None and res_ablation["boundary"] is not None:
        print("\nboundary primary :", {k: round(v, 4) for k, v in zip(metric_names, res_primary["boundary"])})
        print("boundary ablation:", {k: round(v, 4) for k, v in zip(metric_names, res_ablation["boundary"])})
        print("boundary delta(ablation - primary):", {
            k: round(v2 - v1, 4) for k, v1, v2 in zip(metric_names, res_primary["boundary"], res_ablation["boundary"])
        })
    if res_primary["non_boundary"] is not None and res_ablation["non_boundary"] is not None:
        print("\nnon_boundary primary :", {k: round(v, 4) for k, v in zip(metric_names, res_primary["non_boundary"])})
        print("non_boundary ablation:", {k: round(v, 4) for k, v in zip(metric_names, res_ablation["non_boundary"])})
        print("non_boundary delta(ablation - primary):", {
            k: round(v2 - v1, 4) for k, v1, v2 in zip(metric_names, res_primary["non_boundary"], res_ablation["non_boundary"])
        })


if __name__ == "__main__":
    main()
