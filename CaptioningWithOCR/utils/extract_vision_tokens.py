import os
import argparse
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPImageProcessor
import sys
sys.path.append('/home/chenzhixuan/Workspace/DeepSeek-OCR')
from CaptioningWithOCR.vision_embedder import CLIPPatchExtractor

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_path", default="/home/chenzhixuan/Workspace/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/test_images/image.png", type=str)
    p.add_argument("--out_npy", default="/home/chenzhixuan/Workspace/DeepSeek-OCR/CaptioningWithOCR/utils/image.npy", type=str)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    extractor = CLIPPatchExtractor(device=args.device)
    feat = extractor(args.image_path)  # [N, C]
    feat = feat.squeeze(0)

    os.makedirs(os.path.dirname(args.out_npy) or '.', exist_ok=True)
    np.save(args.out_npy, feat.cpu().float().numpy())
    print(f"saved: {args.out_npy} | tokens={feat.size(0)} dim={feat.size(1)}")


if __name__ == "__main__":
    main()


