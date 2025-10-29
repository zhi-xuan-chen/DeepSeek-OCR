import json
from typing import Dict, Any

import torch
from torch.utils.data import Dataset


class Stage1Pairs(Dataset):
    """
    期望 jsonl 行格式：
      {
        "clip_feat": "/abs/path/to/clip_patches.pt",   # [S, Din]
        "ocr_feat":  "/abs/path/to/ocr_tokens.pt"      # [T, Dout]
      }
    可先离线用 DeepSeek-OCR 的视觉编码器把“文字图片”编码为视觉 tokens 保存为 .pt。
    """

    def __init__(self, manifest: str, max_src: int, max_tgt: int):
        self.items = []
        with open(manifest, 'r', encoding='utf-8') as f:
            for line in f:
                self.items.append(json.loads(line))
        self.max_src = max_src
        self.max_tgt = max_tgt

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.items[idx]
        clip_feat = torch.load(rec["clip_feat"])  # [S, Din]
        ocr_feat = torch.load(rec["ocr_feat"])    # [T, Dout]

        # truncate
        clip_feat = clip_feat[: self.max_src]
        ocr_feat = ocr_feat[: self.max_tgt]

        return {
            "src": clip_feat,
            "tgt": ocr_feat,
        }


def stage1_collate(batch):
    # pad to max len in batch
    src_lens = [b["src"].size(0) for b in batch]
    tgt_lens = [b["tgt"].size(0) for b in batch]
    max_src = max(src_lens)
    max_tgt = max(tgt_lens)

    src_dim = batch[0]["src"].size(-1)
    tgt_dim = batch[0]["tgt"].size(-1)

    src_pad = torch.zeros(len(batch), max_src, src_dim, dtype=batch[0]["src"].dtype)
    tgt_pad = torch.zeros(len(batch), max_tgt, tgt_dim, dtype=batch[0]["tgt"].dtype)
    src_mask = torch.ones(len(batch), max_src, dtype=torch.bool)
    tgt_mask = torch.ones(len(batch), max_tgt, dtype=torch.bool)

    for i, rec in enumerate(batch):
        s = rec["src"]
        t = rec["tgt"]
        src_pad[i, : s.size(0)] = s
        tgt_pad[i, : t.size(0)] = t
        src_mask[i, : s.size(0)] = False  # False means valid token
        tgt_mask[i, : t.size(0)] = False

    return {
        "src": src_pad,
        "tgt": tgt_pad,
        "src_key_padding_mask": src_mask,
        "tgt_key_padding_mask": tgt_mask,
    }


