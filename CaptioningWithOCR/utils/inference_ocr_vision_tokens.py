import os
os.environ['HF_HOME'] = '/home/chenzhixuan/.cache/huggingface'

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import sys
sys.path.append("/home/chenzhixuan/Workspace/DeepSeek-OCR/CaptioningWithOCR")
from ocr_vision_embedder import load_backbone_from_pretrained

# 参数
def merge_multimodal_embeddings(input_ids, pre_emb, vision_embeds, post_emb):
    """拼接text和vision embedding，插在<image>对应位置。"""
    return torch.cat([pre_emb, vision_embeds, post_emb], dim=1)

n_embed = 1280
vision_npy = "/home/chenzhixuan/Workspace/DeepSeek-OCR/CaptioningWithOCR/utils/image.npy"
vision_model_id = "deepseek-ai/DeepSeek-OCR"
model_path = "deepseek-ai/DeepSeek-OCR"
prompt = "<image>\nFree OCR."
device = "cuda"

if __name__ == "__main__":
    # 加载vision tokens
    vision_feat = np.load(vision_npy)  # [N, C=1280]
    vision_encoder = load_backbone_from_pretrained(vision_model_id, device="cpu")
    image_newline = vision_encoder.image_newline.detach().cpu()
    view_seperator = vision_encoder.view_seperator.detach().cpu()

    # 拼接vision token + special tokens
    vision = torch.from_numpy(vision_feat).float()
    N, C = vision.shape
    h = w = int(N ** 0.5)
    assert h * w == N, f"N={N} 无法reshape成正方形"
    vision2d = vision.view(h, w, C)
    with_newline = []
    for i in range(h):
        with_newline.append(vision2d[i])
        with_newline.append(image_newline.unsqueeze(0))
    with_newline = torch.cat(with_newline, dim=0)
    vision_with_tokens = torch.cat([with_newline, view_seperator.unsqueeze(0)], dim=0)  # [*, n_embed]
    print(f"vision token shape after concat: {vision_with_tokens.shape}")

    # 加载语言模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval().half().to(device)

    # prompt -> input_ids，手动截断 <image>
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    pos = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    assert len(pos) == 1, "只支持一个<image>token"
    pre = input_ids[0][:pos]
    post = input_ids[0][pos + 1:]
    pre_emb = model.get_input_embeddings()(pre.unsqueeze(0).to(device))
    vision_embeds = vision_with_tokens.unsqueeze(0).to(device).half()
    post_emb = model.get_input_embeddings()(post.unsqueeze(0).to(device))
    full_embeds = merge_multimodal_embeddings(input_ids, pre_emb, vision_embeds, post_emb)
    
    output = super(AutoModel, model).generate(inputs_embeds=full_embeds, max_new_tokens=64)

    # 生成
    with torch.no_grad():
        output = model.generate(inputs_embeds=full_embeds, max_new_tokens=64)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
