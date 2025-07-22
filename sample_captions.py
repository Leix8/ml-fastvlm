import os
import json
import argparse
from PIL import Image
from pathlib import Path

import torch
import pillow_heif  # Enables HEIC support
pillow_heif.register_heif_opener()

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def is_image_file(filename):
    image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.heic')
    return Path(filename.lower()).suffix in image_extensions


def caption_directory(args):
    model_path = os.path.expanduser(args.model_path)
    generation_config = None
    if os.path.exists(os.path.join(model_path, 'generation_config.json')):
        generation_config = os.path.join(model_path, '.generation_config.json')
        os.rename(os.path.join(model_path, 'generation_config.json'), generation_config)

    # Load model
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device=device)
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Prepare prompt
    qs = args.prompt
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    conv_template = conv_templates[args.conv_mode].copy()
    conv_template.append_message(conv_template.roles[0], qs)
    conv_template.append_message(conv_template.roles[1], None)
    prompt = conv_template.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)

    # Enumerate image files recursively
    results = []
    image_dir = os.path.abspath(os.path.expanduser(args.image_dir))
    for root, _, files in os.walk(image_dir):
        for file in files:
            if is_image_file(file):
                image_path = os.path.join(root, file).strip("'\"")
                try:
                    image = Image.open(image_path).convert('RGB')
                except Exception as e:
                    print(f"❌ Skipping {image_path}: {e}")
                    continue

                try:
                    image_tensor = process_images([image], image_processor, model.config)[0].to(device)

                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids,
                            images=image_tensor.unsqueeze(0).half(),
                            image_sizes=[image.size],
                            do_sample=True if args.temperature > 0 else False,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            num_beams=args.num_beams,
                            max_new_tokens=256,
                            use_cache=True
                        )
                        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                        rel_path = os.path.relpath(image_path, start=os.getcwd())
                        print(f"[✓] {rel_path}: {output_text}")
                        results.append({
                            "image_path": rel_path,
                            "caption": output_text
                        })

                except Exception as e:
                    print(f"❌ Failed to process {image_path}: {e}")

    # Save results
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Captions saved to {args.output_json}")

    if generation_config:
        os.rename(generation_config, os.path.join(model_path, 'generation_config.json'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-dir", type=str, required=True, help="Directory containing images (including subfolders)")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to use for captioning")
    parser.add_argument("--output-json", type=str, default="captions.json", help="Path to save captions")
    parser.add_argument("--conv-mode", type=str, default="qwen_2")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    caption_directory(args)