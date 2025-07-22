import os
import json
import argparse
from pathlib import Path

def generate_html(json_file, image_root, output_html):
    with open(json_file, "r") as f:
        data = json.load(f)

    html = [
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>Image Captions</title>",
        "<style>",
        "body { font-family: sans-serif; padding: 20px; }",
        "img { max-width: 500px; display: block; margin-bottom: 10px; }",
        ".entry { margin-bottom: 40px; }",
        ".caption { font-weight: bold; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Image Captions</h1>"
    ]

    image_root = Path(image_root).resolve()

    for item in data:
        image_path = Path(item["image_path"]).resolve()
        if image_root in image_path.parents:
            rel_path = os.path.relpath(image_path, image_root)
        else:
            rel_path = str(image_path)

        if not image_path.exists():
            print(f"⚠️ Warning: Image not found: {image_path}")
            continue

        caption = item["caption"]
        html.append(f"""
        <div class="entry">
            <img src="{image_path.as_posix()}">
            <div class="caption">{caption}</div>
        </div>
        """)

    html.append("</body></html>")

    with open(output_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    print(f"\n✅ HTML file saved: {output_html}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an HTML file to visualize image captions.")
    parser.add_argument("--json-file", type=str, required=True, help="Path to JSON file with image captions.")
    parser.add_argument("--image-root", type=str, required=True, help="Root directory for image paths.")
    parser.add_argument("--output-html", type=str, default="captions.html", help="Output HTML file path.")
    args = parser.parse_args()

    generate_html(args.json_file, args.image_root, args.output_html)
