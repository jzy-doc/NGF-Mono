from __future__ import absolute_import, division, print_function
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from layers import disp_to_depth
from utils import readlines
from options import LiteMonoOptions
import datasets
import networks
import time
from thop import clever_format
from thop import profile


cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")


def profile_once(encoder, decoder, x, edge=None):
    x_e = x[0, :, :, :].unsqueeze(0)
    if edge is not None:
        edge_e = edge[0, :, :, :].unsqueeze(0)
        x_d = encoder(x_e, edge=edge_e)
        flops_e, params_e = profile(encoder, inputs=(x_e, edge_e), verbose=False)
    else:
        x_d = encoder(x_e)
        flops_e, params_e = profile(encoder, inputs=(x_e, ), verbose=False)
    flops_d, params_d = profile(decoder, inputs=(x_d, ), verbose=False)

    flops, params = clever_format([flops_e + flops_d, params_e + params_d], "%.3f")
    flops_e, params_e = clever_format([flops_e, params_e], "%.3f")
    flops_d, params_d = clever_format([flops_d, params_d], "%.3f")

    return flops, params, flops_e, params_e, flops_d, params_d


def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def compute_region_masks(gt_depth, valid_mask, boundary_band_px=1, contour_band_px=2, percentile=90):
    depth = np.log(np.clip(gt_depth, 1e-3, None)).astype(np.float32)

    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)

    valid_grad = grad_mag[valid_mask]
    thr = np.percentile(valid_grad, percentile)
    edge_mask = grad_mag > thr

    if boundary_band_px > 0:
        k = np.ones((2 * boundary_band_px + 1, 2 * boundary_band_px + 1), np.uint8)
        boundary_mask = cv2.dilate(edge_mask.astype(np.uint8), k, iterations=1).astype(bool)
    else:
        boundary_mask = edge_mask

    boundary_mask = boundary_mask & valid_mask
    non_boundary_mask = valid_mask & (~boundary_mask)

    if contour_band_px > 0:
        k2 = np.ones((2 * contour_band_px + 1, 2 * contour_band_px + 1), np.uint8)
        contour_mask = cv2.dilate(edge_mask.astype(np.uint8), k2, iterations=1).astype(bool)
    else:
        contour_mask = edge_mask

    contour_mask = contour_mask & valid_mask
    return boundary_mask, non_boundary_mask, contour_mask


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def build_sobel_edge(rgb, eps=1e-6):
    gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    sobel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0],
          [-2.0, 0.0, 2.0],
          [-1.0, 0.0, 1.0]]],
        device=rgb.device, dtype=rgb.dtype).unsqueeze(0)
    sobel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0],
          [0.0, 0.0, 0.0],
          [1.0, 2.0, 1.0]]],
        device=rgb.device, dtype=rgb.dtype).unsqueeze(0)
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + eps)
    max_val = mag.amax(dim=(2, 3), keepdim=True)
    edge = mag / (max_val + eps)
    return edge


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


def prepare_encoder_input(color, use_edge_input=False, edge_epsilon=1e-6,
                          fourth_channel_mode="edge", fourth_channel_noise_std=0.2,
                          fourth_channel_blur_kernel=5):
    if not use_edge_input:
        return color
    fourth = build_fourth_channel(
        color,
        mode=fourth_channel_mode,
        edge_epsilon=edge_epsilon,
        noise_std=fourth_channel_noise_std,
        blur_kernel=fourth_channel_blur_kernel)
    return torch.cat([color, fourth], dim=1)

def save_confidence_visualization(conf_map, rgb_tensor, save_dir, index):
    os.makedirs(save_dir, exist_ok=True)

    conf_img = (np.clip(conf_map, 0, 1) * 255).astype(np.uint8)
    conf_color = cv2.applyColorMap(conf_img, cv2.COLORMAP_JET)

    rgb_img = np.transpose(rgb_tensor, (1, 2, 0))
    rgb_img = np.clip(rgb_img * 255.0, 0, 255).astype(np.uint8)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    rgb_img = cv2.resize(rgb_img, (conf_color.shape[1], conf_color.shape[0]))

    overlay = cv2.addWeighted(rgb_img, 0.5, conf_color, 0.5, 0)

    cv2.imwrite(os.path.join(save_dir, "{:04d}_rgb.png".format(index)), rgb_img)
    cv2.imwrite(os.path.join(save_dir, "{:04d}_conf.png".format(index)), conf_color)
    cv2.imwrite(os.path.join(save_dir, "{:04d}_overlay.png".format(index)), overlay)

def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    assert not (opt.use_edge_input and opt.use_edge_mid_fusion), \
        "use_edge_input and use_edge_mid_fusion are mutually exclusive"
    if opt.depth_encoder_type == "dino":
        assert not opt.use_edge_input and not opt.use_edge_mid_fusion, \
            "Minimal DINO replacement keeps edge-input/fusion disabled."

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
        confidence_path = os.path.join(opt.load_weights_folder, "confidence.pth")

        encoder_dict = torch.load(encoder_path)
        decoder_dict = torch.load(decoder_path)

        img_ext = '.png' if opt.png else '.jpg'
        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                           [0], 4, is_train=False, img_ext=img_ext)
        dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        if opt.depth_encoder_type == "dino":
            encoder = networks.DINOEncoder(
                model_name=opt.dino_model_name)
        else:
            encoder = networks.LiteMono(
                model=opt.model,
                height=encoder_dict['height'],
                width=encoder_dict['width'],
                in_chans=4 if opt.use_edge_input else 3,
                use_edge_mid_fusion=opt.use_edge_mid_fusion,
                edge_fusion_strength=opt.edge_fusion_strength,
                edge_fusion_strength_mode=opt.edge_fusion_strength_mode,
                edge_fusion_mode=opt.edge_fusion_mode,
                edge_stage_mode=opt.edge_stage_mode,
                # Neuro-inspired fusion options
                use_neuro_fusion=opt.use_neuro_fusion if hasattr(opt, 'use_neuro_fusion') else False,
                neuro_fusion_mode=getattr(opt, 'neuro_fusion_mode', 'gabor'),
                neuro_fusion_stage=getattr(opt, 'neuro_fusion_stage', 'all'),
                neuro_multiscale=getattr(opt, 'neuro_multiscale', True))
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
        model_dict = encoder.state_dict()
        depth_model_dict = depth_decoder.state_dict()
        encoder.load_state_dict({
            k: v for k, v in encoder_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }, strict=False)
        depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_model_dict})

        confidence_decoder = None

        if opt.eval_confidence:
            assert os.path.isfile(confidence_path), \
                "Cannot find confidence head at {}".format(confidence_path)
            confidence_dict = torch.load(confidence_path)

            # infer output channels from saved weights
            conf_out_ch = None
            for k, v in confidence_dict.items():
                if "dispconv" in k and k.endswith("conv.weight"):
                    conf_out_ch = v.shape[0]
                    break
            assert conf_out_ch is not None, "Cannot infer confidence output channels from {}".format(confidence_path)

            confidence_decoder = networks.DepthDecoder(
                encoder.num_ch_enc,
                scales=range(3),
                num_output_channels=conf_out_ch
            )

            confidence_model_dict = confidence_decoder.state_dict()
            confidence_decoder.load_state_dict({
                k: v for k, v in confidence_dict.items()
                if k in confidence_model_dict and v.shape == confidence_model_dict[k].shape
            })
            confidence_decoder.cuda()
            confidence_decoder.eval()

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        pred_disps = []
        pred_confidences = []
        input_rgbs_for_vis = []
        saved_vis_count = 0

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].cuda()

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
                input_encoder = prepare_encoder_input(
                    input_color,
                    use_edge_input=opt.use_edge_input,
                    edge_epsilon=opt.edge_epsilon,
                    fourth_channel_mode=opt.fourth_channel_mode,
                    fourth_channel_noise_std=opt.fourth_channel_noise_std,
                    fourth_channel_blur_kernel=opt.fourth_channel_blur_kernel)

                edge_for_fusion = build_sobel_edge(input_color, eps=opt.edge_epsilon) if opt.use_edge_mid_fusion else None
                flops, params, flops_e, params_e, flops_d, params_d = profile_once(
                    encoder, depth_decoder, input_encoder, edge=edge_for_fusion)
                t1 = time_sync()
                if opt.use_edge_mid_fusion:
                    features = encoder(input_encoder, edge=edge_for_fusion)
                else:
                    features = encoder(input_encoder)

                output = depth_decoder(features)

                confidence_output = None
                if opt.eval_confidence:
                    confidence_output = confidence_decoder(features)
                t2 = time_sync()

                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)
                
                if opt.eval_confidence and confidence_output is not None:
                    conf_np = confidence_output[("disp", 0)].detach().cpu().numpy()  # [B, C, H, W]
                    pred_confidences.append(conf_np)

                    if getattr(opt, "save_confidence_vis", False):
                        rgb_np = input_color.detach().cpu().numpy()
                        vis_dir = opt.confidence_vis_dir if getattr(opt, "confidence_vis_dir", "") else \
                            os.path.join(opt.load_weights_folder, "confidence_vis")

                        for b in range(conf_np.shape[0]):
                            if saved_vis_count >= getattr(opt, "confidence_vis_count", 20):
                                break
                            conf_mean = conf_np[b].mean(axis=0)
                            save_confidence_visualization(conf_mean, rgb_np[b], vis_dir, saved_vis_count)
                            saved_vis_count += 1

        pred_disps = np.concatenate(pred_disps)
        if opt.eval_confidence and len(pred_confidences) > 0:
            pred_confidences = np.concatenate(pred_confidences, axis=0)
        else:
            pred_confidences = None

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    print("-> Evaluating")
    print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []
    boundary_errors = []
    non_boundary_errors = []
    contour_vicinity_errors = []

    confidence_means = []
    confidence_boundary_means = []
    confidence_non_boundary_means = []
    confidence_high_error_means = []
    confidence_low_error_means = []

    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth_full = 1.0 / pred_disp

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([
                0.40810811 * gt_height, 0.99189189 * gt_height,
                0.03594771 * gt_width,  0.96405229 * gt_width
            ]).astype(np.int32)

            crop_mask = np.zeros(mask.shape, dtype=bool)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = True
            mask = np.logical_and(mask, crop_mask)
        else:
            mask = gt_depth > 0

        pred_depth = pred_depth_full[mask]
        gt_depth_valid = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth_valid) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio
        else:
            ratio = 1.0

        pred_depth_full_scaled = pred_depth_full * opt.pred_depth_scale_factor * ratio

        pred_depth = np.clip(pred_depth, MIN_DEPTH, MAX_DEPTH)
        pred_depth_full_scaled = np.clip(pred_depth_full_scaled, MIN_DEPTH, MAX_DEPTH)

        errors.append(compute_errors(gt_depth_valid, pred_depth))

        # region masks: compute once and reuse
        boundary_mask = None
        non_boundary_mask = None
        contour_vicinity_mask = None
        need_region_masks = (
            opt.report_region_metrics or
            (opt.eval_confidence and getattr(opt, "report_confidence_stats", False))
        )

        if need_region_masks:
            boundary_mask, non_boundary_mask, contour_vicinity_mask = compute_region_masks(
                gt_depths[i],
                mask,
                boundary_band_px=opt.boundary_band_px,
                contour_band_px=opt.contour_band_px,
                percentile=90
            )

        # region-wise depth metrics
        if opt.report_region_metrics:
            if boundary_mask is not None and boundary_mask.sum() > 0:
                boundary_errors.append(
                    compute_errors(gt_depths[i][boundary_mask], pred_depth_full_scaled[boundary_mask])
                )
            if non_boundary_mask is not None and non_boundary_mask.sum() > 0:
                non_boundary_errors.append(
                    compute_errors(gt_depths[i][non_boundary_mask], pred_depth_full_scaled[non_boundary_mask])
                )
            if contour_vicinity_mask is not None and contour_vicinity_mask.sum() > 0:
                contour_vicinity_errors.append(
                    compute_errors(gt_depths[i][contour_vicinity_mask], pred_depth_full_scaled[contour_vicinity_mask])
                )

        # confidence statistics
        if opt.eval_confidence and pred_confidences is not None and getattr(opt, "report_confidence_stats", False):
            # pred_confidences[i]: [C, H, W], average over source-frame channels
            conf_map = pred_confidences[i].mean(axis=0)
            conf_map = cv2.resize(conf_map, (gt_width, gt_height))
            conf_map = np.clip(conf_map, 0.0, 1.0)

            if mask.sum() > 0:
                confidence_means.append(conf_map[mask].mean())

            if boundary_mask is not None and boundary_mask.sum() > 0:
                confidence_boundary_means.append(conf_map[boundary_mask].mean())

            if non_boundary_mask is not None and non_boundary_mask.sum() > 0:
                confidence_non_boundary_means.append(conf_map[non_boundary_mask].mean())

            # error-aware confidence statistics
            abs_err_full = np.abs(pred_depth_full_scaled - gt_depths[i])
            valid_err = abs_err_full[mask]

            if valid_err.size > 0:
                err_thr = np.percentile(valid_err, 75)

                high_err_mask = np.logical_and(mask, abs_err_full >= err_thr)
                low_err_mask = np.logical_and(mask, abs_err_full < err_thr)

                if high_err_mask.sum() > 0:
                    confidence_high_error_means.append(conf_map[high_err_mask].mean())
                if low_err_mask.sum() > 0:
                    confidence_low_error_means.append(conf_map[low_err_mask].mean())

    if not opt.disable_median_scaling and len(ratios) > 0:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(
            med, np.std(ratios / med))
        )

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format(
        "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")

    if opt.report_region_metrics:
        print("\n-> Region metrics enabled")
        print("boundary_band_px={}, contour_band_px={}".format(
            opt.boundary_band_px, opt.contour_band_px))

        if len(boundary_errors) > 0:
            boundary_mean = np.array(boundary_errors).mean(0)
            print("\n  [Boundary band]")
            print(("&{: 8.3f}  " * 7).format(*boundary_mean.tolist()) + "\\\\")
        else:
            print("\n  [Boundary band] no valid pixels")

        if len(non_boundary_errors) > 0:
            non_boundary_mean = np.array(non_boundary_errors).mean(0)
            print("\n  [Non-boundary]")
            print(("&{: 8.3f}  " * 7).format(*non_boundary_mean.tolist()) + "\\\\")
        else:
            print("\n  [Non-boundary] no valid pixels")

        if len(contour_vicinity_errors) > 0:
            contour_mean = np.array(contour_vicinity_errors).mean(0)
            print("\n  [Contour vicinity]")
            print(("&{: 8.3f}  " * 7).format(*contour_mean.tolist()) + "\\\\")
        else:
            print("\n  [Contour vicinity] no valid pixels")

    if opt.eval_confidence and getattr(opt, "report_confidence_stats", False):
        print("\n-> Confidence stats")
        if len(confidence_means) > 0:
            print("   full_mean:          {:.4f}".format(np.mean(confidence_means)))
        else:
            print("   full_mean:          no valid pixels")

        if len(confidence_boundary_means) > 0:
            print("   boundary_mean:      {:.4f}".format(np.mean(confidence_boundary_means)))
        else:
            print("   boundary_mean:      no valid pixels")

        if len(confidence_non_boundary_means) > 0:
            print("   non_boundary_mean:  {:.4f}".format(np.mean(confidence_non_boundary_means)))
        else:
            print("   non_boundary_mean:  no valid pixels")

        if len(confidence_high_error_means) > 0:
            print("   high_error_mean:    {:.4f}".format(np.mean(confidence_high_error_means)))
        else:
            print("   high_error_mean:    no valid pixels")

        if len(confidence_low_error_means) > 0:
            print("   low_error_mean:     {:.4f}".format(np.mean(confidence_low_error_means)))
        else:
            print("   low_error_mean:     no valid pixels")

    print("\n  " + (
        "flops: {0}, params: {1}, flops_e: {2}, params_e:{3}, flops_d:{4}, params_d:{5}"
    ).format(flops, params, flops_e, params_e, flops_d, params_d))
    print("\n-> Done!")


if __name__ == "__main__":
    options = LiteMonoOptions()
    evaluate(options.parse())
