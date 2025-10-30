import os
import argparse
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPImageProcessor
import sys
sys.path.append('/home/chenzhixuan/Workspace/DeepSeek-OCR')
from CaptioningWithOCR.vision_embedder import CLIPPatchExtractor
import json
from tqdm import tqdm

class MimicJsonDataset:
    """
    读取 mimic 注解 json，支持 train/validate/test 三个 split，输出所有图片路径和数据标签。
    """
    def __init__(self, json_path, image_root):
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
        return self.samples[idx]

def save_feature_npy(feat: torch.Tensor, out_path: str):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    np.save(out_path, feat.cpu().float().numpy())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_path", type=str, default=None, help="单张图片测试模式")
    p.add_argument("--out_npy", type=str, default=None, help="单张图片时，输出 npy 路径")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--json", type=str, default="/jhcnas5/chenzhixuan/data/mimic_annotations.json", help="mimic 注解 json")
    p.add_argument("--image_root", type=str, default="/jhcnas4/kyle/Xray/DATA/MIMIC-CXR/files", help="mimic 原图根目录")
    p.add_argument("--out_dir", type=str, default="/jhcnas5/chenzhixuan/checkpoints/DeepSeek-OCR/clip_features", help="批量保存的 npy 根目录")
    args = p.parse_args()

    extractor = CLIPPatchExtractor(device=args.device)

    if args.json and args.image_root and args.out_dir:
        dataset = MimicJsonDataset(args.json, args.image_root)
        pbar = tqdm(dataset, desc="提取vision特征", unit="张")
        for rec in pbar:
            out_path = os.path.join(args.out_dir, os.path.splitext(rec['rel_path'])[0] + '.npy')
            if os.path.exists(out_path):
                continue  # 已有则跳过
            try:
                feat = extractor(rec['img_path'])  # [N, C]
                feat = feat.squeeze(0)
                save_feature_npy(feat, out_path)
                pbar.set_postfix({"saved": os.path.basename(out_path), "tokens": feat.size(0)})
            except Exception as e:
                print(f"[ERROR] {rec['img_path']} error: {e}")
        print(f"全部完成，总计 {len(dataset)} 张")
        return

    # 兼容旧版：单图处理
    if args.image_path and args.out_npy:
        feat = extractor(args.image_path)  # [N, C]
        feat = feat.squeeze(0)
        save_feature_npy(feat, args.out_npy)
        print(f"saved: {args.out_npy} | tokens={feat.size(0)} dim={feat.size(1)}")

if __name__ == "__main__":
    main()


