import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import argparse
import torch
import numpy as np
from PIL import Image, features
from transformers import CLIPModel, CLIPImageProcessor
import sys
sys.path.append('/home/chenzhixuan/Workspace/DeepSeek-OCR')
from CaptioningWithOCR.vision_embedder import load_clip_vision
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms as T

class _SimpleImageProcessor:
    """最小可用的图像处理器：Resize->ToTensor->Normalize（单通道）。

    - 将输入 PIL.Image 转为灰度，Resize 到 224
    - 转 tensor 并按均值/方差归一化（mean=0.5, std=0.5）
    - 返回 dict，其中 "pixel_values" 形状为 [1, 1, 224, 224]
    """

    def __init__(self, size: int = 224) -> None:
        self.size = size
        self.transform = T.Compose([
            T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
        ])

    def __call__(self, images: Image.Image):
        if not isinstance(images, Image.Image):
            raise TypeError("images 需要是 PIL.Image")
        pixel = self.transform(images)  # [1, H, W]
        return pixel


class MimicJsonDataset:
    """
    读取 mimic 注解 json，支持 train/validate/test 三个 split，输出所有图片路径和数据标签。
    """
    def __init__(self, json_path, image_root):
        self.transform = _SimpleImageProcessor()
        self.samples = []
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for split in ["train", "validate", "test"]:
            if split in data:
                for item in data[split]:
                    # 标准化字段
                    image_path = item.get("image path") or item.get("image_path")
                    if isinstance(image_path, list):
                        image_path = image_path[0]
                    assert image_path, f"找不到图片路径: {item}"
                    abs_path = os.path.join(image_root, image_path.lstrip("/\\"))
                    self.samples.append({
                        "img_path": abs_path,
                        "rel_path": image_path.lstrip("/\\")
                    })
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path = self.samples[idx]['img_path']
        rel_path = self.samples[idx]['rel_path']
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, rel_path

def collate_fn(batch):
    imgs = [item[0] for item in batch]
    imgs = torch.stack(imgs)
    rel_paths = [item[1] for item in batch]
    return {"img": imgs, "rel_path": rel_paths}

def save_feature_npy(feat: torch.Tensor, out_path: str):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    np.save(out_path, feat.float().numpy())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_root", type=str, default="/jhcnas4/kyle/Xray/DATA/MIMIC-CXR/files", help="mimic 原图根目录")
    p.add_argument("--out_dir", type=str, default="/jhcnas5/chenzhixuan/checkpoints/DeepSeek-OCR/clip_features", help="批量保存的 npy 根目录")
    p.add_argument("--json", type=str, default="/jhcnas5/chenzhixuan/data/mimic_annotations.json", help="mimic 注解 json")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    extractor, device = load_clip_vision(device=args.device)

    dataset = MimicJsonDataset(args.json, args.image_root)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=4)
    pbar = tqdm(dataloader)
    
    for batch in pbar:
        feats = extractor(batch['img'].to(device))  # [N, C]
        feats = feats.detach().cpu().contiguous()
        for i, feat in enumerate(feats):
            out_path = os.path.join(args.out_dir, os.path.splitext(batch['rel_path'][i])[0] + '.npy')
            save_feature_npy(feat, out_path)
            pbar.set_postfix({"saved": os.path.basename(out_path), "tokens": feat.size(0)})
    return

if __name__ == "__main__":
    main()


