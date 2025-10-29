from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase

from process.image_process import DeepseekOCRProcessor


class MultimodalCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        vision_backbone,
        image_token: str = "<image>",
        ignore_index: int = -100,
        max_length: int = 8192,
        train_vision: bool = False,
        crop_mode: bool = True,
        projector_to_lm: Optional[torch.nn.Module] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.vision = vision_backbone
        self.image_token = image_token
        self.ignore_index = ignore_index
        self.max_length = max_length
        self.train_vision = train_vision
        self.crop_mode = crop_mode
        self.projector_to_lm = projector_to_lm
        self.processor = DeepseekOCRProcessor(tokenizer=tokenizer)

        if not train_vision:
            for p in self.vision.parameters():
                p.requires_grad_(False)
            self.vision.eval()

    @torch.no_grad()
    def _encode_text(self, text: str) -> torch.LongTensor:
        return torch.tensor(self.tokenizer.encode(text, add_special_tokens=False), dtype=torch.long)

    def _build_sequence(
        self,
        prompt: str,
        response: str,
        image_embed: torch.Tensor,
        num_img_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # split by <image>
        assert self.image_token in prompt, "prompt must contain <image>"
        parts = prompt.split(self.image_token)
        assert len(parts) == 2, "only support single <image> for now"

        left_ids = self._encode_text(parts[0])
        right_ids = self._encode_text(parts[1])
        resp_ids = self._encode_text(response) + torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)

        image_ids = torch.full((num_img_tokens,), fill_value=self.processor.image_token_id, dtype=torch.long)

        # concat: [bos] left + <image-expanded> + right + response + [eos]
        seq_ids = torch.tensor([self.tokenizer.bos_token_id], dtype=torch.long)
        seq_ids = torch.cat([seq_ids, left_ids, image_ids, right_ids, resp_ids], dim=0)
        seq_ids = seq_ids[: self.max_length]

        # labels: mask everything before response (including image ids & prompt)
        prompt_len = 1 + left_ids.numel() + image_ids.numel() + right_ids.numel()
        labels = seq_ids.clone()
        labels[:prompt_len] = self.ignore_index

        # inputs_embeds: map text ids via LM embedding later; here return positions mask for image block
        image_mask = torch.zeros_like(seq_ids, dtype=torch.bool)
        image_mask[1 + left_ids.numel() : 1 + left_ids.numel() + image_ids.numel()] = True

        return seq_ids, labels, image_mask, image_embed

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_id_list: List[torch.Tensor] = []
        label_list: List[torch.Tensor] = []
        image_mask_list: List[torch.Tensor] = []
        image_embeds_list: List[torch.Tensor] = []

        for item in batch:
            image = item["image"]
            prompt = item["prompt"]
            response = item["response"]

            # compute vision embeddings online
            with torch.set_grad_enabled(self.train_vision):
                vision_seq, num_img_tokens = self.vision(image, crop_mode=self.crop_mode)
                if self.projector_to_lm is not None:
                    vision_seq = self.projector_to_lm(vision_seq)

            seq_ids, labels, image_mask, image_embed = self._build_sequence(
                prompt=prompt, response=response, image_embed=vision_seq, num_img_tokens=num_img_tokens
            )

            input_id_list.append(seq_ids)
            label_list.append(labels)
            image_mask_list.append(image_mask)
            image_embeds_list.append(image_embed)

        input_ids = pad_sequence(input_id_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(label_list, batch_first=True, padding_value=self.ignore_index)
        image_mask = pad_sequence(image_mask_list, batch_first=True, padding_value=False)

        # pack image embeds to a big tensor with per-sample segments
        # build a list of embeds and positions for later substitution
        return {
            "input_ids": input_ids,
            "labels": labels,
            "image_mask": image_mask,
            "image_embeds_list": image_embeds_list,
        }
