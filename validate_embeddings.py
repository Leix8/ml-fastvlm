import argparse
from tensor_similarity import *
import json
import torch
import os
import numpy as np

def load_tensor_from_json(json_dir):
    json_file = json.load(json_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # validation chain: pytorch_linux -> onnx_linux -> dlc_linux -> dlc_android
    parser.add_argument("--pytorch_linux", type = str, required = False, help = "json file, inferenced with pytorch on linux")
    parser.add_argument("--onnx_linux", type = str, required = False, help = "json file, inferenced with onnx on linux")
    parser.add_argument("--dlc_linux", type = str, required = False, help = "raw file, inference with dlc on linux")
    parser.add_argument("--dlc_android", type = str, required = False, help = "raw file, inference with dlc on android")
    parser.add_argument("--output_dir", type = str, required = False, default = "/data01/workspace_leixu/ml-fastvlm/vision_encoder/embeddings", help = "base directory to output result")

    args = parser.parse_args()

    if args.pytorch_linux and args.onnx_linux:
        with open(args.pytorch_linux, "r") as f:
            pytorch_linux_json = json.load(f)
        with open(args.onnx_linux, "r") as f:
            onnx_linux_json = json.load(f)
        
        for image_path, list1 in pytorch_linux_json.items():
            if image_path not in onnx_linux_json:
                continue
            else:
                mode = "pytorch_linux_vs_onnx_linux"
                base_name = os.path.splitext(os.path.basename(image_path))[0] + f"_{mode}.png"
                tensor1 = torch.tensor(list1)
                tensor2 = torch.tensor(onnx_linux_json[image_path])
                cos_sim, l2_dist, scale_diff = compute_similarity(tensor1, tensor2)
                visualize_and_save(image_path = image_path, cos_sim = cos_sim, scale_diff = scale_diff, output_path = os.path.join(args.output_dir, base_name), mode = mode)
    
    if args.onnx_linux and args.dlc_linux:
        with open(args.onnx_linux, "r") as f:
            onnx_linux_json = json.load(f)
        for image_path, onnx_linux_array in onnx_linux_json.items():
            mode = "onnx_linux_vs_dlc_linux"
            base_name = os.path.splitext(os.path.basename(image_path))[0] + f"_{mode}.png"
            onnx_linux_tensor = torch.tensor(onnx_linux_array)

            dlc_linux_array = np.fromfile(args.dlc_linux, dtype=np.float32)
            dlc_linux_tensor = torch.tensor(dlc_linux_array).reshape(onnx_linux_tensor.shape)

            cos_sim, l2_dist, scale_diff = compute_similarity(onnx_linux_tensor, dlc_linux_tensor)
            visualize_and_save(image_path = image_path, cos_sim = cos_sim, scale_diff = scale_diff, output_path = os.path.join(args.output_dir, base_name), mode = mode)
            
    if args.dlc_linux and args.dlc_android:
        mode = "dlc_linux_vs_dlc_android"
        base_name = f"emulation_{mode}.png"
        dlc_linux_array = np.fromfile(args.dlc_linux, dtype=np.float32)
        dlc_android_array = np.fromfile(args.dlc_android, dtype=np.float32)
        print(f"check loaded dlc_linux_array: shape = {dlc_linux_array.shape}, type = {dlc_linux_array.dtype}")
        print(f"check loaded dlc_android_array: shape = {dlc_android_array.shape}, type = {dlc_android_array.dtype}")
        shape = (256, 3072)
        dlc_linux_tensor = torch.tensor(dlc_linux_array).reshape(shape)
        dlc_android_tensor = torch.tensor(dlc_android_array).reshape(shape)
        cos_sim, l2_dist, scale_diff = compute_similarity(dlc_linux_tensor, dlc_android_tensor)
        visualize_and_save(cos_sim = cos_sim, scale_diff = scale_diff, output_path = os.path.join(args.output_dir, base_name), mode = mode)


        