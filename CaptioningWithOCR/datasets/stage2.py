import json
from typing import Dict, Any, List

import torch
from torch.utils.data import Dataset


class Stage2Samples(Dataset):
    """
    期望 jsonl 行格式（可选 ocr_feat 用于对齐损失）：
      {
        "clip_feat": "/abs/path/to/clip_patches.pt",   # [S, Din]
        "prompt": "<image> ...",                       # 必须含 <image>
        "response": "目标caption文本",
        "ocr_feat": "/abs/path/to/ocr_tokens.pt"       # [T, 1280] (可选)
      }
    训练会用 translator 把 clip_feat -> translated_feats (维度 1280)，
    再在 collate 中将 <image> 段展开并用 translated_feats 替换到 inputs_embeds。
    """

    def __init__(self, manifest: str, max_src: int):
        self.items: List[Dict[str, Any]] = []
        with open(manifest, 'r', encoding='utf-8') as f:
            for line in f:
                self.items.append(json.loads(line))
        self.max_src = max_src

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.items[idx]
        clip_feat = torch.load(rec["clip_feat"])  # [S, Din]
        clip_feat = clip_feat[: self.max_src]
        rec_out = {
            "clip_feat": clip_feat,
            "prompt": rec["prompt"],
            "response": rec["response"],
        }

        # optional 对齐目标
        ocr_path = rec.get("ocr_feat")
        if ocr_path:
            try:
                ocr_feat = torch.load(ocr_path)
                rec_out["ocr_feat"] = ocr_feat
            except Exception:
                rec_out["ocr_feat"] = None

        return rec_out


