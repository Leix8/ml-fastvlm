import torch
import argparse
import os
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.utils import disable_torch_init
import onnx 

from module_wrapper import VisionEncoderWrapper, VisionEncoderProjectorWrapper

# load FastVLM model
def load_fastvlm_model(raw_path: str):
    # from transformers import AutoModel
    # model = AutoModel.from_pretrained(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    model_path = os.path.expanduser(raw_path)
    generation_config = None
    if os.path.exists(os.path.join(model_path, 'generation_config.json')):
        generation_config = os.path.join(model_path, '.generation_config.json')
        os.rename(os.path.join(model_path, 'generation_config.json'),
                  generation_config)

    # Load model
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, device = device)
    return model

# extract vision encoder
# class VisionEncoderWrapper(torch.nn.Module):
#     def __init__(self, vision_encoder):
#         super().__init__()
#         self.vision_encoder = vision_encoder
#     def forward(self, x):
#         return self.vision_encoder(x)

# save as .pth and onnx

def find_vision_encoder_projector(model):
    # process: vision encoder only
    vision_encoder  = model.get_vision_tower()
    print(f"vision_encoder found: type = {type(vision_encoder)}")

    # process: projector only
    projector = None
    for name in ["mm_projector", "multi_modal_projector", "vision_proj", "visual_projector", "projector"]:
        if hasattr(model.model, name):
            projector = getattr(model.model, name)
            break
    print(f"projector found: type = {type(projector)},  module: {projector}")
    return vision_encoder, projector

def save_model(vision_encoder_projector, save_dir = "./projector/projector", model_name = "fastvithd", save_onnx = False):
    os.makedirs(save_dir, exist_ok = True)

    pytorch_module_name = model_name + "_vision_encoder_projector.pth"
    torch.save(vision_encoder_projector, f"{save_dir}/{pytorch_module_name}")  # ✅ Save full model
    print(f"Pytorch model saved to {save_dir}/{pytorch_module_name}")

    if save_onnx:
        dtype = next(vision_encoder_projector.parameters()).dtype
        device = next(vision_encoder_projector.parameters()).device
        dummy_input = torch.randn(1, 3, 1024, 1024, dtype=torch.float32).to(device)
        onnx_module_name = model_name + "_vision_encoder_projector.onnx"
        onnx_path = f"{save_dir}/{onnx_module_name}"
        torch.onnx.export(
            vision_encoder_projector, 
            dummy_input,
            onnx_path,
            input_names = ["input"],
            output_names = ["features"],
            opset_version = 12
        )
        print(f"ONNX model has been saved to {onnx_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type = str, required = True, help = "path to FastVLM model")
    parser.add_argument("--save_dir", type = str, default = None, help = "directory to save the extracted encoder")
    parser.add_argument("--save_onnx", action = "store_true", help = "if to export ONNX model")

    args = parser.parse_args()

    model_name = os.path.basename(os.path.normpath(args.model_path))
    
    if not args.save_dir:
        save_dir = os.path.join("./projector/projector", model_name)
    else: 
        save_dir = os.path.join(".", args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    model = load_fastvlm_model(args.model_path)
    # print(f"check vision tower: {dir(model.get_vision_tower()), model.get_vision_tower().input_image_size}")
    # vision_encoder = model.get_vision_tower().vision_tower
    
    # check: model attributes
    # print(f"check all attributes of model.model: {dir(model.model)}")
    # print(f"search for 'projector'")
    # for name, module in model.named_modules():
    #     if "projector" in name.lower():
    #         print(f"checking projector in model attributes: name = {name}, module = {module}")

    vision_encoder, projector = find_vision_encoder_projector(model)
    vision_encoder_projector_wrapper = VisionEncoderProjectorWrapper(vision_encoder, projector)
    
    save_model(vision_encoder_projector_wrapper, save_dir = save_dir, model_name = model_name, save_onnx = args.save_onnx)