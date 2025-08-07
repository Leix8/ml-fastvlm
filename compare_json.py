import argparse
from tensor_similarity import *
import json
import torch
import os

def load_tensor_from_json(json_dir):
    json_file = json.load(json_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json1", type = str, required = True, help = "fisrt json")
    parser.add_argument("--json2", type = str, required = True, help = "second json")
    parser.add_argument("--output_dir", type = str, required = False, default = "/data01/workspace_leixu/ml-fastvlm/vision_encoder/embeddings", help = "base directory to output result")

    args = parser.parse_args()

    with open(args.json1, "r") as f:
        json1 = json.load(f)
    with open(args.json2, "r") as f:
        json2 = json.load(f)
    
    for image_path, list1 in json1.items():
        if image_path not in json2:
            continue
        else:
            base_name = os.path.splitext(os.path.basename(image_path))[0] + ".png"
            print(base_name)
            tensor1 = torch.tensor(list1)
            tensor2 = torch.tensor(json2[image_path])
            cos_sim, l2_dist = compute_similarity(tensor1, tensor2)
            visualize_and_save(cos_sim, l2_dist, os.path.join(args.output_dir, base_name))
            
