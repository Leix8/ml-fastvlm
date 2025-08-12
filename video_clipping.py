
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

    writer = imageio.get_writer(
        output_video_path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p"
    )

    try:
        for i, frame in enumerate(frames):
            fig = plt.figure(figsize=(10, 8), dpi=100)
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 2])

            metric_names = ["cos_sim", "l2_dist", "scale_diff"]
            metric_colors = ["b", "g", "r"]

            # First metric
            ax0 = fig.add_subplot(gs[0])
            ax0.plot(scores[:, 0], color=metric_colors[0], label=metric_names[0])
            ax0.set_ylabel(metric_names[0], color=metric_colors[0])
            ax0.tick_params(axis='y', labelcolor=metric_colors[0])
            ax0.axvline(x=i, color='tomato', linestyle='--', linewidth=1.5)

            # Second metric on right side
            ax1 = ax0.twinx()
            ax1.plot(scores[:, 1], color=metric_colors[1], label=metric_names[1])
            ax1.set_ylabel("", color=metric_colors[1])
            ax1.tick_params(axis='y', labelcolor=metric_colors[1])

            # Third metric as another axis (offset right)
            ax2 = ax0.twinx()
            ax2.spines["right"].set_position(("outward", 30))
            ax2.plot(scores[:, 2], color=metric_colors[2], label=metric_names[2])
            ax2.set_ylabel("magmitude", color=metric_colors[2])
            ax2.tick_params(axis='y', labelcolor=metric_colors[2])

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
            leg = ax0.legend(
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

            fig.subplots_adjust(hspace=0.4)  # 0.4 = more gap, adjust as needed
            ax3 = fig.add_subplot(gs[1])
            ax3.imshow(np.asarray(frame))
            ax3.axis('off')

            fig.canvas.draw()

            # ARGB -> RGB ndarray
            buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            img = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            img = img[:, :, 1:]  # drop A

            # Resize + ensure uint8, even dims
            img = np.asarray(Image.fromarray(img).resize((W, H), Image.BILINEAR), dtype=np.uint8)

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
        output_path = os.path.join(args.output_dir, d.name + ".mp4")
        visualize_and_save_video(frames, stats_array, output_path, image.size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./llava-v1.5-13b", help = "pytorch model path, which is needed for consistent image_processor")
    parser.add_argument("--video_dir", type=str, default=None, help="location of image file")
    parser.add_argument("--frame_dir", type=str, default=None, help="location of image file")
    parser.add_argument("--output_dir", type=str, default="./video_clipping", help="location of image file")

    args = parser.parse_args()
    
    if args.video_dir:
        process_video(args)

    if args.frame_dir:
        process_frame(args)