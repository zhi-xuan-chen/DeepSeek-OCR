import clip
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPModel, CLIPImageProcessor, CLIPVisionConfig
from transformers import SiglipProcessor, SiglipModel, SiglipImageProcessor, SiglipTokenizer
from einops import rearrange, repeat
from timm.models.vision_transformer import Attention, vit_base_patch16_224
import torch
from PIL import Image
from torchvision import transforms as T

class _SimpleImageProcessor:
    """最小可用的图像处理器：Resize->ToTensor->Normalize（单通道）。

    - 将输入 PIL.Image 转为灰度，Resize 到 224
    - 转 tensor 并按均值/方差归一化（mean=0.5, std=0.5）
    - 返回 dict，其中 "pixel_values" 形状为 [1, 1, 224, 224]
    """

    def __init__(self, size: int = 224) -> None:
        self.size = size
        self.transform = T.Compose([
            T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
        ])

    def __call__(self, images: Image.Image):
        if not isinstance(images, Image.Image):
            raise TypeError("images 需要是 PIL.Image")
        pixel = self.transform(images)  # [1, H, W]
        return pixel

class XRCLIP(nn.Module):
    def __init__(self,):
        super().__init__()
        self.model = vit_base_patch16_224(in_chans=1)
        self.model.head = nn.Identity()
        self.model.load_state_dict(torch.load('/jhcnas5/chenzhixuan/checkpoints/VIRAL/XR_clip.ckpt'), strict=True)
        self.model.to(torch.float32)
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:  
        images = images.unsqueeze(0)
        
        if images.shape[1] == 3:
            images = images[:, 0:1, :, :]  # Take first channel and keep dim [B, 1, H, W]
        else:
            images = images

        all_tokens = self.model.forward_features(images) # [B, 197, 768]
        patch_tokens = all_tokens[:, 1:, :] # [B, 768]

        return patch_tokens

def load_clip_vision(device: str = "cuda"):

    device_resolved = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    model = XRCLIP().to(device_resolved)
    model.eval()
    return model, device_resolved


@torch.no_grad()
def extract_clip_patch_tokens(img: Image.Image, model: XRCLIP, processor: _SimpleImageProcessor, device: str = "cuda") -> torch.Tensor:
    """对单张 PIL Image 在线提取视觉 patch token 特征，返回 [N, C] 张量。

    说明：取 vision_model 的 last_hidden_state 去掉 cls（索引 0），仅保留 patch token。
    """
    pixel_values = processor(img)  # 形状 [1, H, W]
    vm_out = model(pixel_values.to(device))
    return vm_out.cpu().contiguous()  # [N, C]


def extract_clip_patch_tokens_from_path(image_path: str, model: XRCLIP, processor: _SimpleImageProcessor, device: str = "cuda") -> torch.Tensor:
    """对图片路径在线提取 patch token 特征，返回 [N, C]。"""
    img = Image.open(image_path).convert("RGB")
    return extract_clip_patch_tokens(img, model, processor, device)


class CLIPPatchExtractor:
    """可复用的在线提取器：训练时常驻内存，反复调用 __call__ 即可。"""

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", device: str = "cuda") -> None:
        self.model, self.device = load_clip_vision(device)
        
        self.processor = _SimpleImageProcessor()

    @torch.no_grad()
    def __call__(self, img_or_path) -> torch.Tensor:
        if isinstance(img_or_path, Image.Image):
            return extract_clip_patch_tokens(img_or_path, self.model, self.processor, self.device)
        elif isinstance(img_or_path, str):
            return extract_clip_patch_tokens_from_path(img_or_path, self.model, self.processor, self.device)
        else:
            raise TypeError("img_or_path 必须是 PIL.Image 或 字符串路径")
