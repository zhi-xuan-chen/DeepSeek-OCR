# 训练说明（在线计算视觉特征 + 可选视觉联合训练）

本脚手架允许：
- 在线从图像计算视觉序列（保持与推理一致的裁剪与token排布）
- 仅LoRA微调文本模型（默认），或通过 `--train_vision true` 联合训练视觉编码器

## 环境

```bash
pip install -r train/requirements-train.txt
```

## 数据格式（JSONL）

```json
{"image_path": "/abs/path/to/img1.png", "prompt": "<image>\n<|grounding|>Convert the document to markdown.", "response": "# 标题\n..."}
```

## 训练命令（在线视觉计算）

```bash
python -m train.train \
  --base_lm_path deepseek-ai/DeepSeek-V2-Lite \
  --tokenizer_path deepseek-ai/DeepSeek-OCR \
  --pretrained_ocr_model_id deepseek-ai/DeepSeek-OCR \
  --train_jsonl /abs/path/to/train.jsonl \
  --output_dir /abs/path/to/output_lora \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
  --num_train_epochs 2 --learning_rate 2e-4 \
  --bf16 true --load_in_8bit true --crop_mode true \
  --train_vision false
```

联合训练视觉编码器（显存更高，速度更慢）：
```bash
python -m train.train \
  --base_lm_path deepseek-ai/DeepSeek-V2-Lite \
  --tokenizer_path deepseek-ai/DeepSeek-OCR \
  --pretrained_ocr_model_id deepseek-ai/DeepSeek-OCR \
  --train_jsonl /abs/path/to/train.jsonl \
  --output_dir /abs/path/to/output_lora_vision \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 32 \
  --num_train_epochs 1 --learning_rate 1e-4 \
  --bf16 true --load_in_8bit false --crop_mode true \
  --train_vision true
```

> 注意：视觉维度需与文本嵌入维度一致（仓库默认视觉输出维度为1280）。若不一致，需要在 `train/collator.py` 中加入线性投影 `projector_to_lm`。
