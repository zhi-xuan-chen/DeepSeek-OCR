import argparse
import os
import json
from PIL import Image, ImageDraw, ImageFont, ImageColor
from tqdm import tqdm


def find_default_chinese_font():
    fonts = [
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "C:\\Windows\\Fonts\\msyh.ttc",
        "C:\\Windows\\Fonts\\simhei.ttf",
        "C:\\Windows\\Fonts\\simsun.ttc",
    ]
    for f in fonts:
        if os.path.exists(f):
            return f
    return None


def find_default_english_font():
    fonts = [
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/System/Library/Fonts/Times.ttc",
        "C:\\Windows\\Fonts\\times.ttf",
    ]
    for f in fonts:
        if os.path.exists(f):
            return f
    return None


def wrap_text(draw, text, font, max_width):
    """按单词或中文字符换行"""
    words = []
    buf = ""
    for ch in text:
        if ch.isspace():
            if buf:
                words.append(buf)
                buf = ""
            words.append(ch)
        elif "\u4e00" <= ch <= "\u9fff":
            if buf:
                words.append(buf)
                buf = ""
            words.append(ch)
        else:
            buf += ch
    if buf:
        words.append(buf)

    lines = []
    current = ""
    for word in words:
        trial = current + word
        bbox = draw.textbbox((0, 0), trial, font=font)
        w = bbox[2] - bbox[0]
        if w <= max_width or not current:
            current = trial
        else:
            lines.append(current.rstrip())
            current = word
    if current:
        lines.append(current.rstrip())
    return lines


def measure_text_block(draw, text, font, width, line_spacing):
    """测量文本块的高度和宽度"""
    lines = wrap_text(draw, text, font, width)
    bbox = draw.textbbox((0, 0), "A", font=font)
    base_line_height = bbox[3] - bbox[1]
    total_height = int(base_line_height * line_spacing * len(lines))
    max_line_width = 0
    for line in lines:
        w = draw.textbbox((0, 0), line, font=font)[2]
        max_line_width = max(max_line_width, w)
    return total_height, max_line_width, base_line_height, lines


def find_best_font_size(draw, text, font_path, width, height, line_spacing, padding=16):
    """二分搜索字体大小，使文字尽量填满但不超出"""
    min_size, max_size = 8, 300
    best_size, best_lines = min_size, []
    best_line_height = 0
    while min_size <= max_size:
        mid = (min_size + max_size) // 2
        font = ImageFont.truetype(font_path, mid)
        total_height, max_line_width, line_height, lines = measure_text_block(draw, text, font, width - 2 * padding, line_spacing)
        if total_height <= height - 2 * padding and max_line_width <= width - 2 * padding:
            best_size = mid
            best_lines = lines
            best_line_height = line_height
            min_size = mid + 1
        else:
            max_size = mid - 1
    return best_size, best_lines, best_line_height



def render_text_image(
    text="The morning sun filtered softly through the blinds, scattering golden light across the quiet room. It was one of those moments that needed no words, just a breath and a pause.",
    width=512,
    height=512,
    bg_color="#FFFFFF",
    text_color="#000000",
    line_spacing=1.5,
):
    """根据文本自动调整字体大小并完全居中"""
    bg_rgb = ImageColor.getrgb(bg_color)
    text_rgb = ImageColor.getrgb(text_color)
    image = Image.new("RGB", (width, height), bg_rgb)
    draw = ImageDraw.Draw(image)

    font_path = find_default_chinese_font() or find_default_english_font()
    if not font_path:
        raise RuntimeError("未找到系统字体，请手动指定字体路径")

    # 自动寻找最佳字体大小
    font_size, lines, line_height = find_best_font_size(draw, text, font_path, width, height, line_spacing)
    font = ImageFont.truetype(font_path, font_size)

    total_height = int(line_height * line_spacing * len(lines))
    y = (height - total_height) // 2  # 垂直居中

    for line in lines:
        w = draw.textbbox((0, 0), line, font=font)[2]
        x = (width - w) // 2  # 水平居中
        draw.text((x, y), line, fill=text_rgb, font=font)
        y += int(line_height * line_spacing)

    return image


def process_dataset(json_path, output_root, width, height, bg_color, text_color, line_spacing=1.5):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    splits = ["train", "validate", "test"]
    total = 0
    ok = 0

    # 预计算总数用于进度条
    for split in splits:
        if split in data and isinstance(data[split], list):
            total += len(data[split])

    pbar = tqdm(total=total, desc="渲染进度", unit="条")

    for split in splits:
        if split not in data or not isinstance(data[split], list):
            continue
        for item in data[split]:
            if not isinstance(item, dict):
                pbar.update(1)
                continue
            report = item.get("report")
            image_path = item.get("image path") or item.get("image_path")
            image_path = image_path[0]

            # 规范化输出路径：output_root + image_path，并将后缀改为 .png
            rel_path = str(image_path).lstrip("/\\")
            out_path = os.path.join(output_root, rel_path)
            out_path = os.path.splitext(out_path)[0] + ".png"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            # remove \n from report
            report = report.replace("\n", "")
            
            img = render_text_image(
                text=report,
                width=width,
                height=height,
                bg_color=bg_color,
                text_color=text_color,
                line_spacing=line_spacing,
            )
            img.save(out_path)
            ok += 1
            pbar.update(1)

    pbar.close()
    print(f"✅ 批量完成：成功 {ok} / 总计 {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="智能文本图片生成（自动字体大小调整 & 防重叠）")
    parser.add_argument("--text", type=str, default="There is no focal consolidation, pleural effusion or pneumothorax. Bilateral nodular opacities that most likely represent nipple shadows. The cardiomediastinal silhouette is normal. Clips project over the left lung, potentially within the breast. The imaged upper abdomen is unremarkable. Chronic deformity of the posterior left sixth and seventh ribs are noted.There is no focal consolidation, pleural effusion or pneumothorax. Bilateral nodular opacities that most likely represent nipple shadows. The cardiomediastinal silhouette is normal. Clips project over the left lung, potentially within the breast. The imaged upper abdomen is unremarkable. Chronic deformity of the posterior left sixth and seventh ribs are noted.There is no focal consolidation, pleural effusion or pneumothorax. Bilateral nodular opacities that most likely represent nipple shadows. The cardiomediastinal silhouette is normal. Clips project over the left lung, potentially within the breast. The imaged upper abdomen is unremarkable. Chronic deformity of the posterior left sixth and seventh ribs are noted.There is no focal consolidation, pleural effusion or pneumothorax. Bilateral nodular opacities that most likely represent nipple shadows. The cardiomediastinal silhouette is normal. Clips project over the left lung, potentially within the breast. The imaged upper abdomen is unremarkable. Chronic deformity of the posterior left sixth and seventh ribs are noted.")
    parser.add_argument("--output", type=str, default="output.jpg")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--bg", type=str, default="#FFFFFF")
    parser.add_argument("--color", type=str, default="#000000")
    parser.add_argument("--json", type=str, default="/jhcnas5/chenzhixuan/data/mimic_annotations.json", help="包含 train/validate/test 的数据集 JSON 路径")
    parser.add_argument("--output_dir", type=str, default="/jhcnas4/kyle/Xray/DATA/MIMIC-CXR/rpimgs", help="批量输出根目录（图片将按 image path 相对路径保存为 .png）")
    parser.add_argument("--line_spacing", type=float, default=1.5, help="行距倍率（默认 1.5）")
    args = parser.parse_args()

    # 批量模式：提供了 --json 与 --output_dir 则执行批量渲染
    if args.json and args.output_dir:
        process_dataset(
            json_path=args.json,
            output_root=args.output_dir,
            width=args.width,
            height=args.height,
            bg_color=args.bg,
            text_color=args.color,
            line_spacing=args.line_spacing,
        )
    else:
        # 单张渲染兼容
        img = render_text_image(
            text=args.text,
            width=args.width,
            height=args.height,
            bg_color=args.bg,
            text_color=args.color,
            line_spacing=args.line_spacing,
        )
        img.save(args.output)
        print(f"✅ 已生成图片：{args.output}")
