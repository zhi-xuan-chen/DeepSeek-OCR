import os


class Stage1Config:
    # data
    train_manifest: str = os.getenv("CWOCR_STAGE1_MANIFEST", "./data/stage1_train.jsonl")
    val_manifest: str = os.getenv("CWOCR_STAGE1_VAL", "./data/stage1_val.jsonl")

    # feature dims
    src_dim: int = int(os.getenv("CWOCR_SRC_DIM", "1024"))   # CLIP patch dim
    tgt_dim: int = int(os.getenv("CWOCR_TGT_DIM", "1280"))   # DeepSeek-OCR vision dim

    # seq lengths
    max_src_tokens: int = int(os.getenv("CWOCR_MAX_SRC", "576"))   # e.g., 24x24
    max_tgt_tokens: int = int(os.getenv("CWOCR_MAX_TGT", "800"))   # depends on OCR layout

    # model
    d_model: int = int(os.getenv("CWOCR_D_MODEL", "1024"))
    nhead: int = int(os.getenv("CWOCR_NHEAD", "16"))
    num_layers: int = int(os.getenv("CWOCR_LAYERS", "6"))
    dim_feedforward: int = int(os.getenv("CWOCR_FF", "4096"))
    dropout: float = float(os.getenv("CWOCR_DROPOUT", "0.1"))
    # translator type: decoder_q | encdec_q
    translator_type: str = os.getenv("CWOCR_TRANS_TYPE", "encdec_q")

    # train
    batch_size: int = int(os.getenv("CWOCR_BS", "2"))
    lr: float = float(os.getenv("CWOCR_LR", "1e-4"))
    wd: float = float(os.getenv("CWOCR_WD", "0.0"))
    max_steps: int = int(os.getenv("CWOCR_MAX_STEPS", "10000"))
    log_every: int = int(os.getenv("CWOCR_LOG_EVERY", "50"))
    save_every: int = int(os.getenv("CWOCR_SAVE_EVERY", "1000"))
    out_dir: str = os.getenv("CWOCR_OUT_DIR", "./outputs/stage1")


class Stage2Config:
    # data
    train_manifest: str = os.getenv("CWOCR_STAGE2_MANIFEST", "./data/stage2_train.jsonl")
    val_manifest: str = os.getenv("CWOCR_STAGE2_VAL", "./data/stage2_val.jsonl")

    # feature dims
    src_dim: int = int(os.getenv("CWOCR_STAGE2_SRC_DIM", "1280"))  # translated to OCR dim

    # ocr
    ocr_model_id: str = os.getenv("CWOCR_OCR_MODEL", "deepseek-ai/DeepSeek-OCR")
    lora_r: int = int(os.getenv("CWOCR_LORA_R", "16"))
    lora_alpha: int = int(os.getenv("CWOCR_LORA_ALPHA", "32"))
    lora_dropout: float = float(os.getenv("CWOCR_LORA_DROPOUT", "0.05"))

    # train
    batch_size: int = int(os.getenv("CWOCR_STAGE2_BS", "1"))
    lr: float = float(os.getenv("CWOCR_STAGE2_LR", "2e-5"))
    max_steps: int = int(os.getenv("CWOCR_STAGE2_STEPS", "5000"))
    out_dir: str = os.getenv("CWOCR_STAGE2_OUT", "./outputs/stage2")

    # translator training and alignment
    train_translator: bool = os.getenv("CWOCR_STAGE2_TRAIN_TRANSLATOR", "false").lower() == "true"
    align_weight: float = float(os.getenv("CWOCR_STAGE2_ALIGN_WEIGHT", "0.0"))  # >0 启用对齐损失
    # translator type: decoder_q | encdec_q (需与Stage1配置一致或兼容)
    translator_type: str = os.getenv("CWOCR_TRANS_TYPE", "encdec_q")


