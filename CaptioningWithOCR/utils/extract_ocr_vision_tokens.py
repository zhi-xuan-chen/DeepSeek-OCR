import os
os.environ['HF_HOME'] = '/home/chenzhixuan/.cache/huggingface'
import sys
import argparse
import torch
import numpy as np
from PIL import Image

# add vllm project path for imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VLLM_DIR = os.path.join(ROOT, "DeepSeek-OCR-master", "DeepSeek-OCR-vllm")
if VLLM_DIR not in sys.path:
    sys.path.insert(0, VLLM_DIR)
    
sys.path.append('/home/chenzhixuan/Workspace/DeepSeek-OCR')
from CaptioningWithOCR.ocr_vision_embedder import load_backbone_from_pretrained

MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'
CROP_MODE = False

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_path", default="/home/chenzhixuan/Workspace/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/test_images/image.png", type=str)
    p.add_argument("--out_npy", default="/home/chenzhixuan/Workspace/DeepSeek-OCR/CaptioningWithOCR/utils/image.npy", type=str)
    p.add_argument("--model_id", type=str, default=MODEL_PATH)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--crop", type=lambda x: x.lower()=="true", default=CROP_MODE)
    args = p.parse_args()

    vb = load_backbone_from_pretrained(args.model_id, device=args.device)
    img = Image.open(args.image_path).convert("RGB")

    with torch.no_grad():
        feat, num_tokens = vb(img, crop_mode=args.crop)  # [T, 1280]
    
    print(f"tokens={num_tokens}")
    print(feat.shape)

    os.makedirs(os.path.dirname(args.out_npy) or '.', exist_ok=True)
    np.save(args.out_npy, feat.cpu().float().numpy())
    print(f"saved: {args.out_npy} | tokens={feat.size(0)} dim={feat.size(1)}")


if __name__ == "__main__":
    main()


