import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4096):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, L, D]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)




class QueryDecoderTranslator(nn.Module):
    """
    基于查询的 decoder：不显式源编码器，直接将 src 投到 d_model 作为 memory，
    用长度=T的查询序列做 cross-attn，输出 [B, T, tgt_dim]。
    需要提供 tgt_mask（或长度）。
    """

    def __init__(
        self,
        src_dim: int,
        tgt_dim: int,
        d_model: int = 1024,
        nhead: int = 16,
        num_layers: int = 6,
        dim_feedforward: int = 4096,
        dropout: float = 0.1,
        max_tgt_len: int = 2048,
    ):
        super().__init__()
        self.src_proj = nn.Linear(src_dim, d_model)
        self.pos_src = PositionalEncoding(d_model, dropout=dropout, max_len=8192)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.query_pos = nn.Embedding(max_tgt_len, d_model)
        self.out_proj = nn.Linear(d_model, tgt_dim)

    def forward(
        self,
        src_tokens: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # src -> memory
        mem = self.src_proj(src_tokens)
        mem = self.pos_src(mem)

        # build queries by length from tgt_mask (False=valid), or use full length of mem
        if tgt_mask is None:
            T = src_tokens.size(1)
            q = self.query_pos.weight[:T].unsqueeze(0).expand(src_tokens.size(0), T, -1)
            q_key_padding_mask = None
        else:
            # per-sample varying length: build max_T and mask
            lengths = (~tgt_mask).sum(dim=1)  # [B]
            max_T = int(lengths.max().item()) if lengths.numel() > 0 else 0
            idx = torch.arange(max_T, device=src_tokens.device)
            q = self.query_pos(idx).unsqueeze(0).expand(src_tokens.size(0), max_T, -1)
            q_key_padding_mask = idx.unsqueeze(0) >= lengths.unsqueeze(1)

        y = self.decoder(
            tgt=q,
            memory=mem,
            tgt_key_padding_mask=q_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        y = self.out_proj(y)
        return y  # [B, T, tgt_dim]


class EncoderDecoderQueryTranslator(nn.Module):
    """
    编码器-解码器 + 查询：先对 src 编码，再用查询解码得到 [B, T, tgt_dim]。
    """

    def __init__(
        self,
        src_dim: int,
        tgt_dim: int,
        d_model: int = 1024,
        nhead: int = 16,
        num_layers: int = 6,
        dim_feedforward: int = 4096,
        dropout: float = 0.1,
        max_tgt_len: int = 2048,
    ):
        super().__init__()
        self.src_proj = nn.Linear(src_dim, d_model)
        self.pos_src = PositionalEncoding(d_model, dropout=dropout, max_len=8192)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        self.query_pos = nn.Embedding(max_tgt_len, d_model)
        self.out_proj = nn.Linear(d_model, tgt_dim)

    def forward(
        self,
        src_tokens: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.src_proj(src_tokens)
        x = self.pos_src(x)
        mem = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        if tgt_mask is None:
            T = src_tokens.size(1)
            q = self.query_pos.weight[:T].unsqueeze(0).expand(src_tokens.size(0), T, -1)
            q_key_padding_mask = None
        else:
            lengths = (~tgt_mask).sum(dim=1)
            max_T = int(lengths.max().item()) if lengths.numel() > 0 else 0
            idx = torch.arange(max_T, device=src_tokens.device)
            q = self.query_pos(idx).unsqueeze(0).expand(src_tokens.size(0), max_T, -1)
            q_key_padding_mask = idx.unsqueeze(0) >= lengths.unsqueeze(1)

        y = self.decoder(
            tgt=q,
            memory=mem,
            tgt_key_padding_mask=q_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        y = self.out_proj(y)
        return y


def build_translator(
    translator_type: str,
    *,
    src_dim: int,
    tgt_dim: int,
    d_model: int = 1024,
    nhead: int = 16,
    num_layers: int = 6,
    dim_feedforward: int = 4096,
    dropout: float = 0.1,
):
    t = translator_type.lower()
    if t == "decoder_q":
        return QueryDecoderTranslator(src_dim, tgt_dim, d_model, nhead, num_layers, dim_feedforward, dropout)
    if t == "encdec_q":
        return EncoderDecoderQueryTranslator(src_dim, tgt_dim, d_model, nhead, num_layers, dim_feedforward, dropout)
    raise ValueError(f"Unknown translator_type: {translator_type}")


