import os
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

from CaptioningWithOCR.configs import Stage2Config
from CaptioningWithOCR.translator import FeatureTranslator
from CaptioningWithOCR.translator import build_translator
from CaptioningWithOCR.datasets.stage2 import Stage2Samples


def build_lm_and_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<｜▁pad▁｜>"})
    lm = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    lm.resize_token_embeddings(len(tokenizer))
    return lm, tokenizer


def cosine_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred = nn.functional.normalize(pred, dim=-1)
    target = nn.functional.normalize(target, dim=-1)
    sim = (pred * target).sum(dim=-1)
    sim = sim.masked_fill(mask, 0.0)
    denom = (~mask).sum().clamp_min(1)
    return 1.0 - sim.sum() / denom


def collate_with_translated_inputs(
    batch: list[Dict[str, Any]], tokenizer, translator: FeatureTranslator, device: str, train_translator: bool
) -> Dict[str, torch.Tensor]:
    # 1) 翻译 CLIP 特征到 OCR 视觉维度
    clip_feats = [b["clip_feat"] for b in batch]  # list of [S, Din]
    max_s = max(x.size(0) for x in clip_feats)
    din = clip_feats[0].size(-1)
    pad_clip = torch.zeros(len(batch), max_s, din)
    mask = torch.ones(len(batch), max_s, dtype=torch.bool)
    for i, x in enumerate(clip_feats):
        pad_clip[i, : x.size(0)] = x
        mask[i, : x.size(0)] = False
    pad_clip = pad_clip.to(device)
    mask = mask.to(device)
    # 构造每条样本的目标长度 = 计划替换的 <image> 段长度
    # 这里与下方构建 image_ids 的长度一致：num_img_tokens = 有效 src token 数
    lengths = (~mask).sum(dim=1)  # [B]
    max_T = int(lengths.max().item()) if lengths.numel() > 0 else 0
    idx = torch.arange(max_T, device=device)
    tgt_mask = idx.unsqueeze(0) >= lengths.unsqueeze(1)  # True=pad

    if train_translator:
        translated = translator(pad_clip, src_key_padding_mask=mask, tgt_mask=tgt_mask)  # [B, T, 1280]
    else:
        with torch.no_grad():
            translated = translator(pad_clip, src_key_padding_mask=mask, tgt_mask=tgt_mask)

    # 2) 组装文本并展开 <image>
    input_id_list = []
    label_list = []
    image_mask_list = []
    ocr_list = []
    ocr_mask_list = []
    for i, rec in enumerate(batch):
        prompt = rec["prompt"]
        response = rec["response"]
        assert "<image>" in prompt, "prompt 必须包含 <image>"
        parts = prompt.split("<image>")
        assert len(parts) == 2, "仅支持单个 <image>"
        left_ids = tokenizer.encode(parts[0], add_special_tokens=False)
        right_ids = tokenizer.encode(parts[1], add_special_tokens=False)
        resp_ids = tokenizer.encode(response, add_special_tokens=False) + [tokenizer.eos_token_id]

        num_img_tokens = int(lengths[i].item())
        image_ids = [tokenizer.convert_tokens_to_ids("<image>")] * num_img_tokens

        seq_ids = [tokenizer.bos_token_id] + left_ids + image_ids + right_ids + resp_ids
        input_id_list.append(torch.tensor(seq_ids, dtype=torch.long))

        prompt_len = 1 + len(left_ids) + len(image_ids) + len(right_ids)
        labels = torch.tensor(seq_ids, dtype=torch.long)
        labels[:prompt_len] = -100
        label_list.append(labels)

        mask_vec = torch.zeros(len(seq_ids), dtype=torch.bool)
        mask_vec[1 + len(left_ids) : 1 + len(left_ids) + len(image_ids)] = True
        image_mask_list.append(mask_vec)

        # optional: 对齐目标
        if rec.get("ocr_feat") is not None and isinstance(rec["ocr_feat"], torch.Tensor):
            o = rec["ocr_feat"]
            ocr_list.append(o)
        else:
            ocr_list.append(None)

    input_ids = nn.utils.rnn.pad_sequence(input_id_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = nn.utils.rnn.pad_sequence(label_list, batch_first=True, padding_value=-100)
    image_mask = nn.utils.rnn.pad_sequence(image_mask_list, batch_first=True, padding_value=False)

    # 3) 返回必要张量；在训练循环里完成 LM embedding 与替换
    # 如果存在 ocr_list，做 padding
    if any(x is not None for x in ocr_list):
        max_t = max((x.size(0) for x in ocr_list if x is not None), default=0)
        if max_t > 0:
            tgt_dim = ocr_list[[i for i,x in enumerate(ocr_list) if x is not None][0]].size(-1)
            o_pad = torch.zeros(len(ocr_list), max_t, tgt_dim)
            o_mask = torch.ones(len(ocr_list), max_t, dtype=torch.bool)
            for i, x in enumerate(ocr_list):
                if x is None:
                    continue
                o_pad[i, : x.size(0)] = x
                o_mask[i, : x.size(0)] = False
            o_pad = o_pad.to(device)
            o_mask = o_mask.to(device)
        else:
            o_pad = None
            o_mask = None
    else:
        o_pad = None
        o_mask = None

    return {
        "input_ids": input_ids.to(device),
        "labels": labels.to(device),
        "image_mask": image_mask.to(device),
        "translated": translated,  # [B, T, 1280]
        "src_mask": mask,          # [B, S] True=pad
        "ocr_tgt": o_pad,          # [B, T, 1280] or None
        "ocr_mask": o_mask,        # [B, T] or None
    }


def train_loop(cfg: Stage2Config, translator_ckpt: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lm, tokenizer = build_lm_and_tokenizer(cfg.ocr_model_id)
    lm = lm.to(device)

    # LoRA for decoder
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )
    lm = get_peft_model(lm, lora)

    # load translator
    ckpt = torch.load(translator_ckpt, map_location="cpu")
    tcfg = ckpt.get("cfg", {})
    translator = build_translator(
        translator_type=cfg.translator_type,
        src_dim=tcfg.get("src_dim", 1024),
        tgt_dim=tcfg.get("tgt_dim", 1280),
        d_model=tcfg.get("d_model", 1024),
        nhead=tcfg.get("nhead", 16),
        num_layers=tcfg.get("num_layers", 6),
        dim_feedforward=tcfg.get("dim_feedforward", 4096),
        dropout=tcfg.get("dropout", 0.1),
    ).to(device)
    translator.load_state_dict(ckpt["model"], strict=True)
    if cfg.train_translator:
        translator.train()
    else:
        translator.eval()

    ds = Stage2Samples(cfg.train_manifest, max_src=4096)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda b: collate_with_translated_inputs(b, tokenizer, translator, device, cfg.train_translator),
    )

    params = list(lm.parameters())
    if cfg.train_translator:
        params += list(translator.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr)

    lm.train()
    step = 0
    while step < cfg.max_steps:
        for batch in dl:
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            image_mask = batch["image_mask"]
            translated = batch["translated"]  # [B, S, 1280]
            src_mask = batch["src_mask"]
            ocr_tgt = batch["ocr_tgt"]
            ocr_mask = batch["ocr_mask"]

            # 文本 token -> 嵌入
            input_embeds = lm.get_input_embeddings()(input_ids)

            # 每条样本：用 translated 的有效部分替换 image 掩码位置
            for i in range(input_ids.size(0)):
                num_pos = int(image_mask[i].sum().item())
                if num_pos == 0:
                    continue
                # 取该样本的有效 translated 长度 = 非pad的 src 长度
                # 这里用 translated 全长的前 num_pos 段做替换
                rep = translated[i, : num_pos]
                input_embeds[i, image_mask[i]] = rep

            outputs = lm(inputs_embeds=input_embeds, labels=labels)
            loss = outputs.loss

            # 可选：加入与 OCR 编码器特征的对齐损失（Stage1 同款）
            if cfg.align_weight > 0.0 and (ocr_tgt is not None):
                # 对齐长度：对每条样本，使用 image_mask 的正样本数作为对齐长度
                # 将 ocr_tgt pad 后的前 num_pos 与 translated 的前 num_pos 对齐
                align_losses = []
                for i in range(input_ids.size(0)):
                    num_pos = int(image_mask[i].sum().item())
                    if num_pos == 0:
                        continue
                    if ocr_tgt is None:
                        continue
                    Ti = min(num_pos, ocr_tgt.size(1))
                    if Ti <= 0:
                        continue
                    pred_i = translated[i, :Ti]
                    tgt_i = ocr_tgt[i, :Ti]
                    mask_i = torch.zeros(Ti, dtype=torch.bool, device=pred_i.device)
                    align_losses.append(cosine_loss(pred_i.unsqueeze(0), tgt_i.unsqueeze(0), mask_i.unsqueeze(0)))
                if len(align_losses) > 0:
                    loss = loss + cfg.align_weight * (sum(align_losses) / len(align_losses))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()

            step += 1
            if step % 50 == 0:
                print(f"stage2 step {step} | loss {loss.item():.4f}")
            if step >= cfg.max_steps:
                break

    os.makedirs(cfg.out_dir, exist_ok=True)
    lm.save_pretrained(cfg.out_dir)


if __name__ == "__main__":
    cfg = Stage2Config()
    # 请将 translator_ckpt 替换为 Stage1 训练生成的 .pt 路径
    translator_ckpt = os.environ.get("CWOCR_TRANSLATOR_CKPT", "./outputs/stage1/translator_step10000.pt")
    train_loop(cfg, translator_ckpt)


