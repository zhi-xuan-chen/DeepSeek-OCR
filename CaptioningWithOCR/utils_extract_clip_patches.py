import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPModel, CLIPImageProcessor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", type=str)
    p.add_argument("out_pt", type=str)
    p.add_argument("--model", type=str, default="openai/clip-vit-large-patch14")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(args.model).to(device)
    model.eval()
    processor = CLIPImageProcessor.from_pretrained(args.model)

    img = Image.open(args.image).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        # vision_model outputs last_hidden_state [B, 1+N, C]; take patch tokens (exclude cls at idx 0)
        vm_out = model.vision_model(pixel_values=pixel_values)
        hidden = vm_out.last_hidden_state[:, 1:, :]  # [1, N, C]

    feat = hidden.squeeze(0).cpu().contiguous()  # [N, C]
    os.makedirs(os.path.dirname(args.out_pt) or '.', exist_ok=True)
    torch.save(feat, args.out_pt)
    print(f"saved: {args.out_pt} | tokens={feat.size(0)} dim={feat.size(1)}")


if __name__ == "__main__":
    main()


