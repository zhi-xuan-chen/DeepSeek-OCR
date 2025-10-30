import os
from typing import Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from CaptioningWithOCR.configs import Stage1Config
from CaptioningWithOCR.translator import build_translator
from CaptioningWithOCR.datasets.stage1 import Stage1Pairs, stage1_collate


def cosine_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred = nn.functional.normalize(pred, dim=-1)
    target = nn.functional.normalize(target, dim=-1)
    sim = (pred * target).sum(dim=-1)
    sim = sim.masked_fill(mask, 0.0)
    denom = (~mask).sum().clamp_min(1)
    return 1.0 - sim.sum() / denom


class Stage1LightningModule(pl.LightningModule):
    def __init__(self, cfg: Stage1Config):
        super().__init__()
        self.save_hyperparameters(vars(cfg))
        self.cfg = cfg
        self.model = build_translator(
            cfg.translator_type,
            src_dim=cfg.src_dim,
            tgt_dim=cfg.tgt_dim,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
        )

    def forward(self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor], tgt_mask: Optional[torch.Tensor]) -> torch.Tensor:
        return self.model(src, src_key_padding_mask=src_key_padding_mask, tgt_mask=tgt_mask)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        src = batch["src"]
        tgt = batch["tgt"]
        src_mask = batch["src_key_padding_mask"]
        tgt_mask = batch["tgt_key_padding_mask"]

        pred = self(src, src_key_padding_mask=src_mask, tgt_mask=tgt_mask)
        assert pred.size(1) == tgt.size(1), "查询式translator应生成与目标等长的序列"
        loss = cosine_loss(pred, tgt, tgt_mask)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=tgt.size(0))
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.wd)
        return opt


class Stage1DataModule(pl.LightningDataModule):
    def __init__(self, cfg: Stage1Config):
        super().__init__()
        self.cfg = cfg
        self.train_ds = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_ds = Stage1Pairs(self.cfg.train_manifest, self.cfg.max_src_tokens, self.cfg.max_tgt_tokens)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=int(os.getenv("CWOCR_NUM_WORKERS", "4")),
            pin_memory=True,
            collate_fn=stage1_collate,
        )


def main():
    cfg = Stage1Config()
    os.makedirs(cfg.out_dir, exist_ok=True)

    # 读取可选训练参数（设备数、精度、梯度累积、策略等）
    devices = int(os.getenv("CWOCR_DEVICES", str(torch.cuda.device_count() if torch.cuda.is_available() else 1)))
    accelerator = os.getenv("CWOCR_ACCELERATOR", "gpu" if torch.cuda.is_available() else "cpu")
    precision = os.getenv("CWOCR_PRECISION", "16-mixed" if torch.cuda.is_available() else "32-true")
    accumulate_grad_batches = int(os.getenv("CWOCR_ACCUM", "1"))
    strategy = os.getenv("CWOCR_STRATEGY", "ddp" if devices > 1 and accelerator == "gpu" else "auto")
    max_steps = cfg.max_steps

    module = Stage1LightningModule(cfg)
    dm = Stage1DataModule(cfg)

    ckpt_cb = ModelCheckpoint(
        dirpath=cfg.out_dir,
        filename="translator-{step}",
        save_top_k=-1,
        save_last=True,
        every_n_train_steps=cfg.save_every,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")
    # WandB 日志配置（通过环境变量可控）
    wandb_project = os.getenv("CWOCR_WANDB_PROJECT", "DeepSeek-OCR-stage1")
    wandb_name = os.getenv("CWOCR_WANDB_NAME", None)
    wandb_entity = os.getenv("CWOCR_WANDB_ENTITY", None)
    wandb_mode = os.getenv("CWOCR_WANDB_MODE", "online")  # 可设为 offline
    os.environ.setdefault("WANDB_MODE", wandb_mode)
    logger = WandbLogger(project=wandb_project, name=wandb_name, entity=wandb_entity, save_dir=cfg.out_dir)

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        max_steps=max_steps,
        log_every_n_steps=cfg.log_every,
        accumulate_grad_batches=accumulate_grad_batches,
        default_root_dir=cfg.out_dir,
        callbacks=[ckpt_cb, lr_cb],
        logger=logger,
    )

    trainer.fit(module, datamodule=dm)


if __name__ == "__main__":
    main()


