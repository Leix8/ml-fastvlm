import os
import json
import argparse
from pathlib import Path
from urllib.parse import quote
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

MAX_WIDTH = 512  # max display width


def convert_and_resize_image(image_path, temp_dir):
    try:
        img = Image.open(image_path)
        img = img.convert("RGB")

        # Resize if too wide
        if img.width > MAX_WIDTH:
            new_height = int(img.height * MAX_WIDTH / img.width)
            img = img.resize((MAX_WIDTH, new_height))

        output_path = Path(temp_dir) / (Path(image_path).stem + ".jpg")
        img.save(output_path, format="JPEG", quality=85)
        return output_path
    except Exception as e:
        print(f"❌ Failed to process {image_path}: {e}")
        return None


def generate_html(json_file, image_root, output_html):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    html = [
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>Image Captions</title>",
        "<style>",
        "body { font-family: sans-serif; padding: 20px; background: #f8f8f8; }",
        ".entry { margin-bottom: 40px; padding: 15px; background: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }",
        ".caption { font-weight: normal; margin-top: 10px; white-space: pre-wrap; line-height: 1.5; }",
        "img { max-width: 100%; height: auto; border: 1px solid #ccc; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>🖼️ Image Captions Viewer</h1>"
    ]

    image_root = Path(image_root).resolve()
    html_dir = Path(output_html).resolve().parent
    temp_dir = html_dir / "__converted_images__"
    temp_dir.mkdir(exist_ok=True)

    for item in data:
        orig_path = (image_root / item["image_path"]).resolve()

        if not orig_path.exists():
            print(f"⚠️ Warning: Image not found: {orig_path}")
            continue

        # Convert HEIC or resize large image
        if orig_path.suffix.lower() == ".heic":
            processed_path = convert_and_resize_image(orig_path, temp_dir)
        else:
            processed_path = convert_and_resize_image(orig_path, temp_dir)

        if not processed_path or not processed_path.exists():
            print(f"⚠️ Skipping image: {orig_path}")
            continue

        try:
            rel_path = os.path.relpath(processed_path, html_dir)
            img_src = quote(rel_path)

            caption = item["caption"].strip()

            html.append(f"""
            <div class="entry">
                <img src="{img_src}" alt="{processed_path.name}">
                <div class="caption">{caption}</div>
            </div>
            """)
        except Exception as e:
            print(f"❌ Error rendering image {orig_path}: {e}")
            continue

    html.append("</body></html>")

    with open(output_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    print(f"\n✅ HTML file saved: {output_html}")
    print(f"📁 Converted/resized images saved in: {temp_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an HTML file to visualize image captions.")
    parser.add_argument("--json-file", type=str, required=True, help="Path to JSON file with image captions.")
    parser.add_argument("--image-root", type=str, required=True, help="Root directory for image paths.")
    parser.add_argument("--output-html", type=str, default="captions.html", help="Output HTML file path.")
    args = parser.parse_args()

    generate_html(args.json_file, args.image_root, args.output_html)
