# -*- coding: utf-8 -*-
"""
Windows 安全加载 API Key + BMP 转 PNG 再上传 版本
================================================
改动点：
1. 不再硬编码 API_KEY：
   - 优先顺序：环境变量 OPENROUTER_API_KEY > .env 文件（若存在）> 运行时交互输入（隐藏输入，不回显）。
   - 推荐：在项目根目录创建 .env 写入：OPENROUTER_API_KEY=sk-or-xxxxxxx
2. 支持普通图片格式，同时对 .bmp 做内存中转为 PNG（不写临时文件）后再 base64 编码上传。
   - 其它格式 (png/jpg/jpeg/webp/gif) 直接读取原文件。
   - BMP 会在内存转成 PNG，因此上传给模型的是 PNG Data URL。
3. 仍使用 OpenRouter chat/completions 接口，发送文本 + 图像（data URL）。
4. 若返回内容中找不到 base64 图片，保存原始 JSON 以便排查。
5. 保留 tqdm 进度条。
6. 目录结构：
      input_images/   放你的原始图片 (含 .bmp)
      output_results/ 脚本自动生成
      .env            (可选) 放 API key

依赖：
    pip install requests tqdm pillow python-dotenv

运行：
    python generate_images_windows_secure.py

根据需要修改下方“可配置区域”的 PROMPT / MODEL / 输入输出目录等。

安全说明：
- 不要把 .env 或任何包含真实 Key 的文件提交到公共仓库。
- 如果使用交互输入，脚本不会回显你的 key，进程退出后不会保存。

如果模型不返回生成图像（可能只是图像理解），请换成支持生成的模型，并根据实际返回结构调整 extract_base64_images。
"""

import os
import time
import json
import base64
import mimetypes
from pathlib import Path
from typing import List, Dict, Any
import requests
from tqdm import tqdm
from io import BytesIO
from getpass import getpass

try:
    # 可选加载 .env
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from PIL import Image
except ImportError as e:
    raise SystemExit("未安装 Pillow，请先运行: pip install pillow") from e


# ================== 可配置区域（按需修改） ==================
MODEL = "google/gemini-2.5-flash-image-preview:free"

INPUT_DIR = r".\input"
OUTPUT_DIR = r".\output"

# 输出保存的文件后缀（即使 BMP 也会转生成 PNG）
OUTPUT_EXT = ".png"

PROMPT = (
    "Generate a semantic segmentation mask for the golden wires in this image. The background should retain its original pixels. Each golden wire's body should be uniformly red (RGB: 255, 0, 0), precisely segmenting only the wire's main body. The pins of each golden wire should be uniformly green (RGB: 0, 255, 0). Ensure strict RGB color consistency for all segmented parts."
)

# 可选 Header（统计用，可留空）
OPTIONAL_HEADERS = {
    # "HTTP-Referer": "https://your-site.example",
    # "X-Title": "YourProjectName"
}

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MAX_RETRIES = 3
TIMEOUT_SECONDS = 120
BACKOFF_BASE = 2.0

# 支持的输入图片扩展（包含 bmp）
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
# ============================================================


def load_api_key() -> str:
    """
    优先从环境变量 / .env 读取；若无则交互输入（隐藏回显）。
    """
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    print("未检测到环境变量 OPENROUTER_API_KEY。")
    key = getpass("请手动输入你的 OpenRouter API Key (输入时不显示): ").strip()
    if not key:
        raise ValueError("未提供 API Key，无法继续。")
    return key


def list_image_files(input_dir: Path) -> List[Path]:
    return [
        p for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


def guess_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "image/png"


def load_and_convert_if_needed(path: Path) -> bytes:
    """
    若是 BMP，转换为 PNG (内存中)；其他格式直接读二进制。
    返回：bytes (最终要编码的图像数据，且都将按 PNG 处理 BMP 的情形)
    """
    suffix = path.suffix.lower()
    if suffix == ".bmp":
        # 转 PNG
        with Image.open(path) as img:
            # 确保格式安全
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGBA")
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            return buffer.getvalue()
    else:
        # 直接读取原始文件
        with open(path, "rb") as f:
            return f.read()


def build_data_url(path: Path) -> str:
    """
    根据文件后缀判断 MIME。
    如果是 BMP，我们转换成 PNG，所以 MIME 强制设为 image/png。
    """
    if path.suffix.lower() == ".bmp":
        mime = "image/png"
        raw = load_and_convert_if_needed(path)
    else:
        raw = load_and_convert_if_needed(path)
        # 注意：如果是 gif/webp 等，直接 guess；也可以强行转 png（看需求）
        if path.suffix.lower() in (".gif", ".webp"):
            # 有些模型可能对动图 / WebP 支持一般，必要时也可以统一转 PNG：
            # 这里按“原格式”处理，如果你想统一转 PNG，直接复用 BMP 的逻辑即可。
            mime = guess_mime_type(path)
        else:
            mime = guess_mime_type(path)

    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def build_messages(prompt_template: str, image_path: Path, data_url: str) -> List[Dict[str, Any]]:
    final_prompt = prompt_template.replace("{filename}", image_path.name)
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": final_prompt},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]
        }
    ]


def request_with_retries(payload: dict, headers: dict) -> dict:
    attempt = 0
    last_err = None
    while attempt < MAX_RETRIES:
        try:
            resp = requests.post(
                OPENROUTER_ENDPOINT,
                headers=headers,
                data=json.dumps(payload),
                timeout=TIMEOUT_SECONDS
            )
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            else:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        except Exception as e:
            last_err = e
        attempt += 1
        if attempt < MAX_RETRIES:
            time.sleep(BACKOFF_BASE * attempt)
    raise RuntimeError(f"请求失败（已重试 {MAX_RETRIES} 次）: {last_err}")


def extract_base64_images(resp_json: dict) -> List[str]:
    """
    尝试从多种可能结构中提取 base64 图片（不带或带 data: 前缀都接收）。
    支持结构：
      1) choices[*].message.content = list[...] 其中的 dict 可能含:
         - "image_base64"
         - "b64_json"
         - {"type": "image"/"output_image", "data": "..."}
         - {"type": "image_url", "image_url": {"url": "data:image/png;base64,...."}}
      2) choices[*].message.images = list[...] 其中 item.image_url.url = data URL
      3) choices[*].message.content 是字符串且里面含 data:image（可按需补正则）
    """
    results: List[str] = []
    choices = resp_json.get("choices", [])
    if not choices:
        return results

    for ch in choices:
        msg = ch.get("message", {})
        # 先处理 images 数组（你的案例在这里）
        images_arr = msg.get("images")
        if isinstance(images_arr, list):
            for item in images_arr:
                if not isinstance(item, dict):
                    continue
                img_url_obj = item.get("image_url")
                if isinstance(img_url_obj, dict):
                    url = img_url_obj.get("url")
                    if isinstance(url, str) and "base64" in url:
                        results.append(url)  # 可能是 data:image/png;base64,...
        # 处理 content 列表
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                # 常见字段
                if "image_base64" in part and isinstance(part["image_base64"], str):
                    results.append(part["image_base64"])
                elif "b64_json" in part and isinstance(part["b64_json"], str):
                    results.append(part["b64_json"])
                elif part.get("type") in ("image", "output_image") and isinstance(part.get("data"), str):
                    results.append(part["data"])
                elif part.get("type") == "image_url":
                    iu = part.get("image_url")
                    if isinstance(iu, dict):
                        url = iu.get("url")
                        if isinstance(url, str) and "base64" in url:
                            results.append(url)
        elif isinstance(content, str):
            # 可选：简单探测字符串里是否包含 data:image （一般很少）
            if "data:image" in content and "base64" in content:
                # 这里可以用正则提取多段，这里简单直接全部加入
                results.append(content)

    return results


def save_base64_images(b64_list: List[str], out_dir: Path, base_name: str, timestamp: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for idx, b64_data in enumerate(b64_list, start=1):
        original = b64_data
        if b64_data.startswith("data:"):
            try:
                b64_data = b64_data.split(",", 1)[1]
            except Exception:
                b64_data = original
        try:
            raw = base64.b64decode(b64_data)
        except Exception:
            continue
        filename = f"{base_name}_{timestamp}_{idx}{OUTPUT_EXT}"
        (out_dir / filename).write_bytes(raw)
        count += 1
    return count


def process_single_image(img_path: Path, output_root: Path, headers: dict):
    data_url = build_data_url(img_path)
    messages = build_messages(PROMPT, img_path, data_url)
    payload = {
        "model": MODEL,
        "messages": messages,
        # 可加 "temperature": 0.7 等
    }
    resp_json = request_with_retries(payload, headers)
    base64_imgs = extract_base64_images(resp_json)
    timestamp = time.strftime("%H%M%S", time.localtime())
    if base64_imgs:
        out_subdir = output_root / img_path.stem
        saved = save_base64_images(base64_imgs, out_subdir, img_path.stem, timestamp)
        return {"status": "ok", "saved": saved}
    else:
        # 保存原始 JSON
        raw_path = output_root / f"{img_path.stem}_raw_{timestamp}.json"
        raw_path.write_text(json.dumps(resp_json, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"status": "no_image", "saved": 0}


def main():
    print("=== 批量调用 OpenRouter (安全加载 Key + BMP 转 PNG) ===")
    try:
        api_key = load_api_key()
    except Exception as e:
        print(f"[错误] {e}")
        return

    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"[错误] 输入目录不存在: {input_dir}")
        return

    images = list_image_files(input_dir)
    if not images:
        print("[提示] 未找到任何图片（支持: png/jpg/jpeg/webp/bmp/gif）。")
        return

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    headers.update(OPTIONAL_HEADERS)

    print(f"发现 {len(images)} 张图片，开始处理...\n")

    success = 0
    no_image = 0
    failed = 0

    for img_path in tqdm(images, desc="Processing", unit="img"):
        try:
            result = process_single_image(img_path, output_dir, headers)
            if result["status"] == "ok":
                success += 1
            elif result["status"] == "no_image":
                no_image += 1
        except Exception as e:
            failed += 1
            err_log = output_dir / "error.log"
            with err_log.open("a", encoding="utf-8") as ef:
                ef.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {img_path.name}: {repr(e)}\n")

    print("\n=== 处理完成 ===")
    print(f"成功保存图片的请求 : {success}")
    print(f"未返回图片 (仅保存原始JSON): {no_image}")
    print(f"失败                : {failed}")
    print(f"输出目录            : {output_dir.resolve()}")
    if no_image > 0:
        print("提示：若模型本身不生成图片，请换支持生成的模型并调整 extract_base64_images。")


if __name__ == "__main__":
    main()