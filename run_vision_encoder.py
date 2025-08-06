
import os
import argparse
import json

import torch
import torch.serialization

import onnxruntime as ort
import numpy as np

from PIL import Image
from vision_encoder_wrapper import VisionEncoderWrapper
from pathlib import Path
from tqdm import tqdm
# torch.serialization.add_safe_globals([VisionEncoderWrapper])
# torch.serialization.add_safe_globals([MobileCLIPVisionTower])
# from llava.utils import disable_torch_init
# from llava.conversation import conv_templates
# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

def is_image_file(filename):
    image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.heic')
    return Path(filename.lower()).suffix in image_extensions

def pytorch_predict(args):
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
            outputs = vision_tower(image_tensor)
            all_embeddings = outputs.squeeze(0)
            print(f"check outputs shape: {outputs.shape}")
            all_embeddings_list = all_embeddings.cpu().tolist()
            embeddings_dict[image_path] = all_embeddings_list

    json_filename = f"{model_name}_on_{image_dir_name}_pytorch.json"
    with open(os.path.join("vision_encoder", json_filename), "w") as f:
        json.dump(embeddings_dict, f)
        print(f"image embedding json file has been saved to: {json_filename}")

def get_onnx_model_path(pth_model_path):
    base_path, _ = os.path.splitext(pth_model_path)
    return base_path + ".onnx"

def onnx_predict(args):
    pytorch_model_path = os.path.expanduser(args.model_path) # .pth model path
    model_name= os.path.splitext(os.path.basename(pytorch_model_path))[0]  # "llava-fastvithd_0.5b_stage3_encoder"
    onnx_model_path = get_onnx_model_path(pytorch_model_path)
    print(f"get onnx model_path: {onnx_model_path}")

    #load pytorch model for image processor
    vision_encoder_wrapper = torch.load(pytorch_model_path, weights_only=False)
    vision_encoder_wrapper.eval()

    vision_encoder = vision_encoder_wrapper.vision_encoder
    # vision_tower = vision_encoder.to("cuda" if torch.cuda.is_available() else "cpu")
    image_processor = vision_encoder.image_processor

    #load onnx model
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_model_path, providers=providers)

    input_name = session.get_inputs()[0].name
    print(f"✅ ONNX input name: {input_name}")

    # vision_encoder = vision_encoder_wrapper.vision_encoder
    # vision_tower = vision_encoder.to("cuda" if torch.cuda.is_available() else "cpu")
    # image_processor = vision_encoder.image_processor

    # device = next(vision_tower.parameters()).device

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
        image_tensor = image_processor(image, return_tensors="pt")["pixel_values"]  # [1, 3, H, W]

        # image_tensor = image_processor(image, return_tensors='pt')['pixel_values'].to(device) # [1, 3, H, W]
        image_numpy = image_tensor.numpy()
        output = session.run(None, {input_name: image_numpy})[0]  # output: [1, N, D] or [1, D, H, W]
        # output = output.squeeze(0)  # [N, D] or [D, H, W]
        print(f"check outputs shape: {output.shape}")
        embeddings_dict[image_path] = output.tolist()
        # with torch.no_grad():
        #     outputs = vision_tower(image_tensor)
        #     print(f"check outputs shape: {outputs.shape}")
        #     all_embeddings = outputs.squeeze(0)
        #     all_embeddings_list = all_embeddings.cpu().tolist()
        #     embeddings_dict[image_path] = all_embeddings_list

    json_filename = f"{model_name}_on_{image_dir_name}_onnx.json"
    with open(os.path.join("vision_encoder", json_filename), "w") as f:
        json.dump(embeddings_dict, f)
        print(f"image embedding json file has been saved to: {json_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./llava-v1.5-13b", help = "pytorch model path, which is needed for consistent image_processor")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None, help="location of image file")
    parser.add_argument("--run_onnx", action = "store_true", help = "if to run on ONNX model")
    # parser.add_argument("--prompt", type=str, default="Describe the image.", help="Prompt for VLM.")
    # parser.add_argument("--conv-mode", type=str, default="qwen_2")
    # parser.add_argument("--temperature", type=float, default=0.2)
    # parser.add_argument("--top_p", type=float, default=None)
    # parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()
    if args.run_onnx:
        onnx_predict(args)
    else:
        pytorch_predict(args)
