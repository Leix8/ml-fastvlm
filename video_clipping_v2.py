
import os
import argparse
import json
from pathlib import Path
import re
from tqdm import tqdm
import imageio  # pip install imageio imageio-ffmpeg

import torch
import torch.serialization

import numpy as np

# import cv2
from PIL import Image
from vision_encoder_wrapper import VisionEncoderWrapper
from tensor_similarity import *

def is_image_file(filename):
    image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.heic')
    return Path(filename.lower()).suffix in image_extensions

def is_video_file(filename):
    video_extensions = ('.mp4', ".mov", ".avi")
    return Path(filename.lower()).suffix in video_extensions

def natural_key(path: Path):
    """Natural sort: splits 'frame_10.png' into ['frame_', 10, '.png'] so 2 < 10."""
    s = path.name
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def robust_z(x, eps=1e-9):
    x = np.asarray(x, dtype=np.float32)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return (x - med) / (1.4826 * (mad + eps))

def moving_avg(x, k=7):
    if k <= 1: return x
    k = int(k)
    pad = k // 2
    xpad = np.pad(x, (pad, pad), mode='edge')
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(xpad, kernel, mode='valid')

def pick_peaks_strongest_first(change, thr, min_dist, refine_win=2):
    """
    change   : (N,) fused change score
    thr      : threshold; consider only candidates > thr
    min_dist : minimum spacing between selected peaks (in frames)
    refine_win: (optional) snap each chosen index to the local argmax within ±refine_win

    returns: sorted list of peak indices
    """
    change = np.asarray(change)
    N = change.size

    # 1) candidates above threshold
    cand = np.flatnonzero(change > thr)
    if cand.size == 0:
        return []

    # 2) order by strength (descending score)
    order = cand[np.argsort(change[cand])[::-1]]

    # 3) greedy selection with non-maximum suppression
    taken = []
    suppressed = np.zeros(N, dtype=bool)

    for i0 in order:
        if suppressed[i0]:
            continue

        # (optional) refine to nearest local maximum in a small window
        if refine_win > 0:
            lo = max(0, i0 - refine_win)
            hi = min(N, i0 + refine_win + 1)
            i = lo + np.argmax(change[lo:hi])
        else:
            i = i0

        taken.append(i)

        # suppress a neighborhood of size min_dist around i
        L = max(0, i - (min_dist - 1))
        R = min(N, i + min_dist)       # exclusive
        suppressed[L:R] = True

    taken = sorted(set(taken))
    # keep peaks strictly inside (0, N) if you need that invariant:
    taken = [i for i in taken if 0 < i < N]

    return taken

def segment_from_stats(
    stats, 
    fps = 10,
    weights=(0.5, 0.25, 0.25),
    smooth_win=7,
    k_thresh=3.0,
    min_clip_sec=2.0
):
    """
    stats: (n,3) array: [cos_sim, l2_norm_diff, l2_norm_diff_pct]
    returns: dict with change score, cut indices, and clip ranges (frames & seconds)
    """
    stats = np.asarray(stats, dtype=np.float32)
    assert stats.ndim == 2 and stats.shape[1] == 3, "stats must be (n,3)"
    n = stats.shape[0]

    cos_sim   = stats[:, 0]
    l2_diff   = stats[:, 1]
    l2_pct    = stats[:, 2]

    # 1) similarity -> dissimilarity in [0,1]
    d_cos = np.clip((1.0 - cos_sim) / 2.0, 0, 1)

    # 2) robust z-normalize each signal
    z_cos  = robust_z(d_cos)
    z_l2   = robust_z(l2_diff)
    z_lpct = robust_z(l2_pct)

    # 3) smooth
    z_cos  = moving_avg(z_cos,  smooth_win)
    z_l2   = moving_avg(z_l2,   smooth_win)
    z_lpct = moving_avg(z_lpct, smooth_win)

    # 4) fuse
    w0, w1, w2 = weights
    change = w0*z_cos + w1*z_l2 + w2*z_lpct

    # 5) adaptive threshold using robust stats
    c_med = np.median(change)
    c_mad = np.median(np.abs(change - c_med))
    thr = c_med + k_thresh * 1.4826 * (c_mad + 1e-9)

    # peak picking with minimum distance (in frames)
    min_clip_frames = max(1, int(min_clip_sec * fps))

    #order based peak detection
    # peaks = []
    # last = -min_clip_frames
    # for i in range(1, n-1):
    #     if i - last < min_clip_frames:
    #         continue
    #     if change[i] > thr and change[i] >= change[i-1] and change[i] >= change[i+1]:
    #         peaks.append(i)
    #         last = i

    # # 6) turn cuts into clips
    # cuts = sorted(peaks)
    # # ensure cuts are within (0, n-1)
    # cuts = [c for c in cuts if 0 < c < n]

    # score based peak detection:
    cuts = pick_peaks_strongest_first(change, thr, min_clip_frames, refine_win=2)

    # build [start, end) frame ranges
    edges = [0] + cuts + [n]
    clips_frames = [(edges[i], edges[i+1]) for i in range(len(edges)-1)]
    clips_secs   = [ (s/fps, e/fps) for (s,e) in clips_frames ]

    return {
        "clip_score": change,
        "threshold": thr,
        "cuts": cuts,
        "clips_frames": clips_frames,
        "clips_seconds": clips_secs,
    }

def process_video(args): #TBD
    model_path = os.path.expanduser(args.model_path)
    model_name= os.path.splitext(os.path.basename(model_path))[0]  # "llava-fastvithd_0.5b_stage3_encoder"
    print(f"get model_path: {model_path}")

    vision_encoder_wrapper = torch.load(model_path, weights_only=False)
    vision_encoder_wrapper.eval()

    vision_encoder = vision_encoder_wrapper.vision_encoder
    vision_tower = vision_encoder.to("cuda" if torch.cuda.is_available() else "cpu")
    image_processor = vision_encoder.image_processor

    device = next(vision_tower.parameters()).device

    image_dir = os.path.abspath(os.path.expanduser(args.image_dir))
    print(f"get image dir: {image_dir}")
    image_dir_name = os.path.basename(os.path.normpath(image_dir))  # "test"

    image_paths = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if is_image_file(file):
                image_path = os.path.join(root, file).strip("'\"")
                image_paths.append(image_path)

    embeddings_dict = {}

    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"❌ Skipping {image_path}: {e}")
            continue

        image_tensor = image_processor(image, return_tensors='pt')['pixel_values'].to(device)
        with torch.no_grad():
            output = vision_tower(image_tensor)
            all_embeddings = output.squeeze(0)
            print(f"check output type: {type(all_embeddings),} shape: {all_embeddings.shape}")
            all_embeddings_list = all_embeddings.cpu().tolist()
            embeddings_dict[image_path] = all_embeddings_list

    json_filename = f"{model_name}_on_{image_dir_name}_pytorch.json"
    json_path = os.path.join("vision_encoder", json_filename)
    with open(json_path, "w") as f:
        json.dump(embeddings_dict, f)
        print(f"image embedding json file has been saved to: {json_path}")

def iter_frame_dirs(
    root_dir,
    img_exts={".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".HEIC"},
    min_images=2,
    include_root=True,
):
    root = Path(root_dir)
    dirs = [root] if include_root else []
    dirs += [p for p in root.rglob("*") if p.is_dir()]

    seen = set()  # avoid duplicates if symlinks/etc.
    for d in dirs:
        if d in seen:
            continue
        seen.add(d)

        # Only images directly in this directory (do not recurse here)
        imgs = [p for p in d.iterdir()
                if p.is_file()
                and not p.name.startswith(".")
                and p.suffix.lower() in img_exts]
        # print(f"imgs={imgs}")
        if len(imgs) >= min_images:
            imgs.sort(key=natural_key)  # natural numeric order
            # print(d, imgs)
            yield d, imgs

def visualize_and_save_video(frames, scores, output_video_path, img_size, fps=10):
    H, W = img_size

    # yuv420p needs even dimensions
    H -= (H % 2)
    W -= (W % 2)
    scaling_factor = 10  # pixels per point
    W_dynamic = min(max(W, len(scores) * scaling_factor), 1920)  # Cap at 1920 px

    writer = imageio.get_writer(
        output_video_path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p"
    )
    n = scores.shape[0]                       # number of frames/points

    # --- Adaptive figure width (inches) ---
    # Aim for ~80 points per inch; clamp to [10, 28] inches for speed.
    pts_per_inch = 30.0
    fig_w = max(30.0, min(50.0, n / pts_per_inch))
    fig_h = 8.0

    try:
        for i, frame in enumerate(frames):
            fig = plt.figure(figsize=(10, 10), dpi=100)  # wider when n is large
            gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 2])

            metric_names = ["cos_sim", "l2_mag", "l2_mag_%"]
            metric_colors = ["b", "g", "r"]

            # # First metric
            # for idx in range(len(metric_names)):
            #     ax = fig.add_subplot(gs[idx])
            #     ax.plot(scores[:, idx], color=metric_colors[idx], label=metric_names[idx])
            #     ax.set_ylabel(metric_names[idx], color=metric_colors[idx])
            #     ax.tick_params(axis='y', labelcolor=metric_colors[idx])
            #     ax.axvline(x=i, color='tomato', linestyle='--', linewidth=1.5)
            #     ax.legend(loc="upper right")   # <- per-axes legend

            
            ax0 = fig.add_subplot(gs[0])
            ax0.plot(scores[:, 0], color=metric_colors[0], label=metric_names[0])
            ax0.set_ylabel(metric_names[0], color=metric_colors[0])
            ax0.tick_params(axis='y', labelcolor=metric_colors[0])
            ax0.axvline(x=i, color='tomato', linestyle='--', linewidth=1.5)

            ax1 = fig.add_subplot(gs[1])
            ax1.plot(scores[:, 1], color=metric_colors[1], label=metric_names[1])
            ax1.set_ylabel(metric_names[1], color=metric_colors[1])
            ax1.tick_params(axis='y', labelcolor=metric_colors[1])
            ax1.axvline(x=i, color='tomato', linestyle='--', linewidth=1.5)

            ax2 = fig.add_subplot(gs[2])
            ax2.plot(scores[:, 2], color=metric_colors[2], label=metric_names[2])
            ax2.set_ylabel(metric_names[2], color=metric_colors[2])
            ax2.tick_params(axis='y', labelcolor=metric_colors[2])
            ax2.axvline(x=i, color='tomato', linestyle='--', linewidth=1.5)

            # Title and legend
            ax0.set_title(f"Metrics Progression (Frame {i}/{len(frames)-1})")

            lines, labels = [], []
            for ax in (ax0, ax1, ax2):
                h, l = ax.get_legend_handles_labels()
                lines += h; labels += l

            # (optional) dedupe labels while preserving order
            seen = set()
            uniq = [(h, l) for h, l in zip(lines, labels) if not (l in seen or seen.add(l))]
            lines, labels = zip(*uniq)

            # place legend below the plot, horizontal, 3 columns
            leg = ax2.legend(
                lines, labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),   # center below the axes
                ncol=3,                        # 3 items per row
                frameon=False,
                columnspacing=1.5,
                handlelength=2.0
            )

            # make room at the bottom so it doesn't get cut off
            plt.subplots_adjust(bottom=0.25)   # tweak as needed
            # or: plt.tight_layout(rect=[0, 0.1, 1, 1])

            fig.subplots_adjust(hspace=1)  # 0.4 = more gap, adjust as needed
            ax3 = fig.add_subplot(gs[3])
            ax3.imshow(np.asarray(frame))
            ax3.axis('off')

            fig.canvas.draw()

            # ARGB -> RGB ndarray
            buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            img = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            img = img[:, :, 1:]  # drop A

            # Resize + ensure uint8, even dims
            img = np.asarray(Image.fromarray(img).resize((W_dynamic, H), Image.BILINEAR), dtype=np.uint8)

            writer.append_data(img)  # HxWx3, uint8 RGB
            plt.close(fig)
    finally:
        writer.close()
def visualize_and_clip_video(frames, scores, clip_stats, output_video_path, img_size, fps=10):
    H, W = img_size

    # yuv420p needs even dimensions
    H -= (H % 2)
    W -= (W % 2)
    scaling_factor = 10  # pixels per point
    W_dynamic = min(max(W, len(scores) * scaling_factor), 1920)  # Cap at 1920 px

    writer = imageio.get_writer(
        output_video_path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p"
    )
    n = scores.shape[0]                       # number of frames/points

    # --- Adaptive figure width (inches) ---
    # Aim for ~80 points per inch; clamp to [10, 28] inches for speed.
    pts_per_inch = 30.0
    fig_w = max(30.0, min(50.0, n / pts_per_inch))
    fig_h = 8.0

    try:
        for i, frame in enumerate(frames):
            fig = plt.figure(figsize=(10, 10), dpi=100)  # wider when n is large
            gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 1, 2])

            metric_names = ["cos_sim", "l2_mag", "l2_mag_%", "composite_score"]
            metric_colors = ["b", "g", "r", "y"]

            ax0 = fig.add_subplot(gs[0])
            ax0.plot(scores[:, 0], color=metric_colors[0], label=metric_names[0])
            ax0.set_ylabel(metric_names[0], color=metric_colors[0])
            ax0.tick_params(axis='y', labelcolor=metric_colors[0])
            ax0.axvline(x=i, color='tomato', linestyle='--', linewidth=1.5)

            ax1 = fig.add_subplot(gs[1])
            ax1.plot(scores[:, 1], color=metric_colors[1], label=metric_names[1])
            ax1.set_ylabel(metric_names[1], color=metric_colors[1])
            ax1.tick_params(axis='y', labelcolor=metric_colors[1])
            ax1.axvline(x=i, color='tomato', linestyle='--', linewidth=1.5)

            ax2 = fig.add_subplot(gs[2])
            ax2.plot(scores[:, 2], color=metric_colors[2], label=metric_names[2])
            ax2.set_ylabel(metric_names[2], color=metric_colors[2])
            ax2.tick_params(axis='y', labelcolor=metric_colors[2])
            ax2.axvline(x=i, color='tomato', linestyle='--', linewidth=1.5)

            ax3 = fig.add_subplot(gs[3])
            ax3.plot(clip_stats["clip_score"], color=metric_colors[3], label=metric_names[3])
            ax3.set_ylabel(metric_names[3], color=metric_colors[3])
            ax3.tick_params(axis='y', labelcolor=metric_colors[3])
            ax3.axvline(x=i, color='tomato', linestyle='--', linewidth=1.5)

            # Title and legend
            ax0.set_title(f"Metrics Progression (Frame {i}/{len(frames)-1})")

            lines, labels = [], []
            for ax in (ax0, ax1, ax2, ax3):
                h, l = ax.get_legend_handles_labels()
                lines += h; labels += l

            # (optional) dedupe labels while preserving order
            seen = set()
            uniq = [(h, l) for h, l in zip(lines, labels) if not (l in seen or seen.add(l))]
            lines, labels = zip(*uniq)

            # place legend below the plot, horizontal, 3 columns
            leg = ax3.legend(
                lines, labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),   # center below the axes
                ncol=4,                        # 3 items per row
                frameon=False,
                columnspacing=1.5,
                handlelength=2.0
            )

            # make room at the bottom so it doesn't get cut off
            plt.subplots_adjust(bottom=0.25)   # tweak as needed
            # or: plt.tight_layout(rect=[0, 0.1, 1, 1])

            fig.subplots_adjust(hspace=1)  # 0.4 = more gap, adjust as needed
            ax4 = fig.add_subplot(gs[4])
            ax4.imshow(np.asarray(frame))
            ax4.axis('off')

            fig.canvas.draw()

            # ARGB -> RGB ndarray
            buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            img = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            img = img[:, :, 1:]  # drop A

            # Resize + ensure uint8, even dims
            img = np.asarray(Image.fromarray(img).resize((W_dynamic, H), Image.BILINEAR), dtype=np.uint8)

            writer.append_data(img)  # HxWx3, uint8 RGB
            plt.close(fig)
    finally:
        writer.close()

def process_frame(args):
    model_path = os.path.expanduser(args.model_path)
    model_name= os.path.splitext(os.path.basename(model_path))[0]  # "llava-fastvithd_0.5b_stage3_encoder"
    print(f"get model_path: {model_path}")

    vision_encoder_wrapper = torch.load(model_path, weights_only=False)
    vision_encoder_wrapper.eval()

    vision_encoder = vision_encoder_wrapper.vision_encoder
    vision_tower = vision_encoder.to("cuda" if torch.cuda.is_available() else "cpu")
    image_processor = vision_encoder.image_processor

    device = next(vision_tower.parameters()).device

    for d, image_dirs in iter_frame_dirs(root_dir=args.frame_dir): # processing frames of one same video        
        pre_embedding = None
        stats = []
        frames = []
        for idx, image_dir in tqdm(enumerate(image_dirs), total=len(image_dirs), desc=f"Processing {d}"): # processing one frame
            try:
                image = Image.open(image_dir).convert('RGB')
            except Exception as e:
                print(f"❌ Skipping {image_dir}: {e}")
                continue    
            
            image_tensor = image_processor(image, return_tensors='pt')['pixel_values'].to(device)
            
            with torch.no_grad():
                output = vision_tower(image_tensor)
                cur_embedding = output.squeeze(0)
                if pre_embedding != None: 
                    cos_sim, l2_dist, scale_diff = compute_similarity(pre_embedding.reshape(-1, 1), cur_embedding.reshape(-1, 1), dim = 0)
                    frame_stat = [
                        float(np.squeeze(cos_sim)),
                        float(np.squeeze(l2_dist)),
                        float(np.squeeze(scale_diff))
                    ]
                    stats.append(frame_stat)
                    frames.append(image)
                pre_embedding = cur_embedding
        
        stats_array = np.array(stats)
        if args.clip:
            # segment_from_stats -> return {"change_score": change, "threshold": thr, "cuts": cuts, "clips_frames": clips_frames, "clips_seconds": clips_secs,}
            clip_stats = segment_from_stats(stats)
            output_path = os.path.join(args.output_dir, d.name + "_clipped.mp4")
            visualize_and_clip_video(frames, stats_array, clip_stats, output_path, image.size)
        else:
            output_path = os.path.join(args.output_dir, d.name + "_unclipped.mp4")
            visualize_and_save_video(frames, stats_array, output_path, image.size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./llava-v1.5-13b", help = "pytorch model path, which is needed for consistent image_processor")
    parser.add_argument("--video_dir", type=str, default=None, help="location of image file")
    parser.add_argument("--frame_dir", type=str, default=None, help="location of image file")
    parser.add_argument("--output_dir", type=str, default="./video_clipping", help="location of image file")
    parser.add_argument("--clip", action = "store_true", help = "if to execute clipping")

    args = parser.parse_args()
    
    if args.video_dir:
        process_video(args)

    if args.frame_dir:
        process_frame(args)