import os
from typing import Tuple, List

import torch
import torch.nn as nn
from PIL import Image, ImageOps

from deepencoder.sam_vary_sdpa import build_sam_vit_b
from deepencoder.clip_sdpa import build_clip_l
from deepencoder.build_linear import MlpProjector
from addict import Dict

from process.image_process import DeepseekOCRProcessor, dynamic_preprocess
from config import IMAGE_SIZE, BASE_SIZE


class VisionBackbone(nn.Module):
    def __init__(self, n_embed: int = 1280):
        super().__init__()
        self.sam_model = build_sam_vit_b()
        self.vision_model = build_clip_l()
        self.projector = MlpProjector(Dict(projector_type="linear", input_dim=2048, n_embed=n_embed))
        # special tokens following deepseek_ocr.py
        embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
        self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std)
        self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)

    @torch.no_grad()
    def forward(self, image: Image.Image, crop_mode: bool = True) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            embeddings: [num_img_tokens, n_embed]
            num_tokens: int
        """
        device = next(self.parameters()).device
        processor = DeepseekOCRProcessor()

        # compute crops consistent with processor
        if image.size[0] <= 640 and image.size[1] <= 640:
            crop_ratio = [1, 1]
            images_crop_raw = []
        else:
            if crop_mode:
                images_crop_raw, crop_ratio = dynamic_preprocess(image, image_size=IMAGE_SIZE)
            else:
                crop_ratio = [1, 1]
                images_crop_raw = []

        # global view
        global_view = ImageOps.pad(image, (BASE_SIZE, BASE_SIZE),
                                   color=tuple(int(x * 255) for x in processor.image_transform.mean))
        global_tensor = processor.image_transform(global_view).unsqueeze(0).to(device)

        # local views
        local_tensors: List[torch.Tensor] = []
        if images_crop_raw:
            for im in images_crop_raw:
                local_tensors.append(processor.image_transform(im))
        if len(local_tensors) > 0:
            local_tensor = torch.stack(local_tensors, dim=0).to(device)
        else:
            local_tensor = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), device=device)

        # encode
        local_features_1 = self.sam_model(local_tensor)
        local_features_2 = self.vision_model(local_tensor, local_features_1)
        local_features = torch.cat((local_features_2[:, 1:], local_features_1.flatten(2).permute(0, 2, 1)), dim=-1)
        local_features = self.projector(local_features)

        global_features_1 = self.sam_model(global_tensor)
        global_features_2 = self.vision_model(global_tensor, global_features_1)
        global_features = torch.cat((global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)), dim=-1)
        global_features = self.projector(global_features)

        # layout to 2D + separators
        _, hw, n_dim = global_features.shape
        h = w = int(hw ** 0.5)
        num_width_tiles, num_height_tiles = crop_ratio

        global_features = global_features.view(h, w, n_dim)
        global_features = torch.cat(
            [global_features, self.image_newline[None, None, :].expand(h, 1, n_dim)], dim=1
        )
        global_features = global_features.view(-1, n_dim)

        if local_tensor.sum() != 0 and (num_width_tiles > 1 or num_height_tiles > 1):
            _2, hw2, n_dim2 = local_features.shape
            h2 = w2 = int(hw2 ** 0.5)
            local_features = local_features.view(num_height_tiles, h2, num_width_tiles, w2, n_dim2) \
                                         .permute(0, 2, 1, 3, 4).reshape(num_height_tiles * h2, num_width_tiles * w2, n_dim2)
            local_features = torch.cat(
                [local_features, self.image_newline[None, None, :].expand(num_height_tiles * h2, 1, n_dim2)], dim=1
            ).view(-1, n_dim2)
            vision_seq = torch.cat([local_features, global_features, self.view_seperator[None, :]], dim=0)
        else:
            vision_seq = torch.cat([global_features, self.view_seperator[None, :]], dim=0)

        # num image tokens consistent with processor.tokenize_with_images
        num_queries = (IMAGE_SIZE // 16 + (1 if IMAGE_SIZE % 16 else 0)) // 4
        num_queries_base = (BASE_SIZE // 16 + (1 if BASE_SIZE % 16 else 0)) // 4
        num_img_tokens = (num_queries_base + 1) * num_queries_base + 1
        if num_width_tiles > 1 or num_height_tiles > 1:
            num_img_tokens += ((num_queries * num_width_tiles) + 1) * (num_queries * num_height_tiles)

        assert vision_seq.shape[0] == num_img_tokens, f"vision tokens {vision_seq.shape[0]} != expected {num_img_tokens}"
        return vision_seq, num_img_tokens


def load_backbone_from_pretrained(model_id: str, device: str = "cuda") -> VisionBackbone:
    # We only need parts of weights from deepseek-ai/DeepSeek-OCR state dict
    model = VisionBackbone().to(device)
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    # try to load safetensors
    local_dir = snapshot_download(model_id)
    # find model*.safetensors
    candidates = [p for p in os.listdir(local_dir) if p.endswith(".safetensors")]
    if len(candidates) == 0:
        raise FileNotFoundError("No .safetensors found in pretrained repo")
    state = load_file(os.path.join(local_dir, candidates[0]))

    with torch.no_grad():
        for name, param in model.named_parameters():
            # map names similar to deepseek_ocr.load_weights but for encoder parts
            # expected prefixes as in original model: model.sam_model, model.vision_model, model.projector, model.image_newline, model.view_seperator
            key = f"model.{name}"
            if key in state:
                param.copy_(state[key].to(param.device).to(param.dtype))
    model.eval()
    return model
