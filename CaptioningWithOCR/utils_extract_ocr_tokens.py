import os
import sys
import argparse
import torch
from PIL import Image

# add vllm project path for imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VLLM_DIR = os.path.join(ROOT, "DeepSeek-OCR-master", "DeepSeek-OCR-vllm")
if VLLM_DIR not in sys.path:
    sys.path.insert(0, VLLM_DIR)

from vision_embedder import load_backbone_from_pretrained

MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'
CROP_MODE = False

def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", type=str)
    p.add_argument("out_pt", type=str)
    p.add_argument("--model_id", type=str, default=MODEL_PATH)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--crop", type=lambda x: x.lower()=="true", default=CROP_MODE)
    args = p.parse_args()

    vb = load_backbone_from_pretrained(args.model_id, device=args.device)
    img = Image.open(args.image).convert("RGB")

    with torch.no_grad():
        feat, _ = vb(img, crop_mode=args.crop)  # [T, 1280]

    os.makedirs(os.path.dirname(args.out_pt) or '.', exist_ok=True)
    torch.save(feat.cpu(), args.out_pt)
    print(f"saved: {args.out_pt} | tokens={feat.size(0)} dim={feat.size(1)}")


if __name__ == "__main__":
    main()


