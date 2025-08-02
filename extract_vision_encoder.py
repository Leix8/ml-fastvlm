import torch
import argparse
import os
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.utils import disable_torch_init
import onnx 

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
class VisionEncoderWrapper(torch.nn.Module):
    def __init__(self, vision_encoder):
        super().__init__()
        self.vision_encoder = vision_encoder

    def forward(self, x):
        return self.vision_encoder(x)

# save as .pth and onnx
def save_model(encoder, save_dir = "vision_encoder", model_name = "fastvithd", save_onnx = False):
    os.makedirs(save_dir, exist_ok = True)

    pytorch_encoder_name = model_name + "_encoder.pth"
    torch.save(encoder.state_dict(), f"{save_dir}/{pytorch_encoder_name}")
    print(f"Pytorch weights saved to {save_dir}/{pytorch_encoder_name}")

    if save_onnx:
        dtype = next(encoder.parameters()).dtype
        device = next(encoder.parameters()).device
        # llava-fastvithd_0.5b_stage3: image_processor.crop_size = (1024, 1024)
        dummy_input = torch.randn(1, 3, 1024, 1024, dtype=dtype).to(device)
        onnx_encoder_name = model_name + "_encoder.onnx"
        onnx_path = f"{save_dir}/{onnx_encoder_name}"
        torch.onnx.export(
            encoder, 
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
        save_dir = os.path.join("vision_encoder", model_name)
    else: 
        save_dir = os.path.join("vision_encoder", args.save_dir)

    model = load_fastvlm_model(args.model_path)
    # print(f"check vision tower: {dir(model.get_vision_tower()), model.get_vision_tower().input_image_size}")
    vision_encoder = model.get_vision_tower().vision_tower
    print(f"check type of vision_encoder: {type(vision_encoder)}")
    encoder_wrapper = VisionEncoderWrapper(vision_encoder)

    save_model(encoder_wrapper, save_dir = save_dir, model_name = model_name, save_onnx = args.save_onnx)