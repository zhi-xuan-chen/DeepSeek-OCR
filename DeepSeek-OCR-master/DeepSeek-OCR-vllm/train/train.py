import os
import argparse
from typing import Dict, Any

import torch
from torch.utils.data import Dataset
from PIL import Image

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

from train.dataset import JsonlDataset
from train.vision_embedder import load_backbone_from_pretrained
from train.collator import MultimodalCollator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", type=str, default="/abs/path/to/train.jsonl")
    p.add_argument("--output_dir", type=str, default="/abs/path/to/output_lora")
    p.add_argument("--base_lm_path", type=str, default="deepseek-ai/DeepSeek-V2-Lite")
    p.add_argument("--tokenizer_path", type=str, default="deepseek-ai/DeepSeek-OCR")
    p.add_argument("--pretrained_ocr_model_id", type=str, default="deepseek-ai/DeepSeek-OCR")

    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--max_length", type=int, default=8192)

    p.add_argument("--bf16", type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--fp16", type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--load_in_8bit", type=lambda x: x.lower() == "true", default=True)

    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    p.add_argument("--train_vision", type=lambda x: x.lower() == "true", default=False)
    p.add_argument("--crop_mode", type=lambda x: x.lower() == "true", default=True)

    return p.parse_args()


class WrappedDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.ds = JsonlDataset(jsonl_path)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.ds[idx]
        img = Image.open(s.image_path).convert("RGB")
        return {
            "image": img,
            "prompt": s.prompt,
            "response": s.response,
        }


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<｜▁pad▁｜>"})

    # load base LM
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    device_map = "auto" if torch.cuda.is_available() else None

    if args.load_in_8bit:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        lm = AutoModelForCausalLM.from_pretrained(
            args.base_lm_path, trust_remote_code=True, device_map=device_map, quantization_config=quant_config
        )
    else:
        lm = AutoModelForCausalLM.from_pretrained(
            args.base_lm_path, trust_remote_code=True, torch_dtype=dtype, device_map=device_map
        )

    lm.resize_token_embeddings(len(tokenizer))

    # apply LoRA
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )
    lm = get_peft_model(lm, lora)

    # build vision backbone (online) from DeepSeek-OCR weights
    vision = load_backbone_from_pretrained(args.pretrained_ocr_model_id, device="cuda" if torch.cuda.is_available() else "cpu")

    # collator: will compute image features online and build inputs_embeds later in data_collator
    collator = MultimodalCollator(
        tokenizer=tokenizer,
        vision_backbone=vision,
        image_token="<image>",
        ignore_index=-100,
        max_length=args.max_length,
        train_vision=args.train_vision,
        crop_mode=args.crop_mode,
        projector_to_lm=None,
    )

    # wrap dataset
    train_ds = WrappedDataset(args.train_jsonl)

    def data_collator(features: list[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = collator(features)
        # substitute image masks with embeds to inputs_embeds
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        image_mask = batch["image_mask"]
        image_embeds_list = batch["image_embeds_list"]

        # map text token ids to embeds
        input_embeds = lm.get_input_embeddings()(input_ids.to(lm.device))

        # substitute per-sample image spans with embeds
        for i in range(input_ids.size(0)):
            mask = image_mask[i]
            num_pos = int(mask.sum().item())
            if num_pos == 0:
                continue
            image_embeds = image_embeds_list[i].to(input_embeds.dtype).to(input_embeds.device)
            image_embeds = image_embeds[:num_pos]
            input_embeds[i, mask] = image_embeds

        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        return {
            "inputs_embeds": input_embeds,
            "labels": labels.to(lm.device),
            "attention_mask": attention_mask.to(lm.device),
        }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=10,
        save_steps=1000,
        save_strategy="steps",
        bf16=args.bf16,
        fp16=args.fp16,
        remove_unused_columns=False,
        report_to=["none"],
    )

    trainer = Trainer(
        model=lm,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
