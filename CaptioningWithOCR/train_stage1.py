import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from CaptioningWithOCR.configs import Stage1Config
from CaptioningWithOCR.translator import build_translator
from CaptioningWithOCR.datasets.stage1 import Stage1Pairs, stage1_collate


def cosine_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # pred/target: [B, T, D], mask: [B, T] (True=pad)
    pred = nn.functional.normalize(pred, dim=-1)
    target = nn.functional.normalize(target, dim=-1)
    sim = (pred * target).sum(dim=-1)  # [B, T]
    sim = sim.masked_fill(mask, 0.0)
    denom = (~mask).sum().clamp_min(1)
    return 1.0 - sim.sum() / denom


def train_loop(cfg: Stage1Config) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)

    ds = Stage1Pairs(cfg.train_manifest, cfg.max_src_tokens, cfg.max_tgt_tokens)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, collate_fn=stage1_collate)

    model = build_translator(
        cfg.translator_type,
        src_dim=cfg.src_dim,
        tgt_dim=cfg.tgt_dim,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
    ).cuda()

    # encoder 类型已移除，默认使用查询式 decoder

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    global_step = 0
    model.train()
    while global_step < cfg.max_steps:
        for batch in dl:
            src = batch["src"].cuda(non_blocking=True)
            tgt = batch["tgt"].cuda(non_blocking=True)
            src_mask = batch["src_key_padding_mask"].cuda(non_blocking=True)
            tgt_mask = batch["tgt_key_padding_mask"].cuda(non_blocking=True)

            # 查询式：使用 tgt_mask 指定目标长度，输出应与 tgt 等长
            pred = model(src, src_key_padding_mask=src_mask, tgt_mask=tgt_mask)
            assert pred.size(1) == tgt.size(1), "查询式translator应生成与目标等长的序列"
            loss = cosine_loss(pred, tgt, tgt_mask)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            global_step += 1
            if global_step % cfg.log_every == 0:
                print(f"step {global_step} | loss {loss.item():.4f}")

            if global_step % cfg.save_every == 0:
                ckpt = {
                    "model": model.state_dict(),
                    "step": global_step,
                    "cfg": vars(cfg),
                }
                torch.save(ckpt, os.path.join(cfg.out_dir, f"translator_step{global_step}.pt"))

            if global_step >= cfg.max_steps:
                break


if __name__ == "__main__":
    cfg = Stage1Config()
    train_loop(cfg)


