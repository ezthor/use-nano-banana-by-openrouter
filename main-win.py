# -*- coding: utf-8 -*-
"""

依赖安装：
    pip install requests tqdm pillow python-dotenv

运行：
    python main_win_simple_interrupt.py

可按需修改下方“配置区域”变量，如 MODEL / INPUT_DIR / PROMPT 等。
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

# 可选：加载 .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from PIL import Image
except ImportError as e:
    raise SystemExit("未安装 Pillow，请先运行: pip install pillow") from e


# ================== 配置区域 ==================
MODEL = "google/gemini-2.5-flash-image-preview:free"

INPUT_DIR = r".\input"
OUTPUT_DIR = r".\output"

OUTPUT_EXT = ".png"  # 统一输出后缀

PROMPT = """

Objective: From the provided grayscale image, create a clean, colored tracing of the bond wires and pads. This is a technical tracing task, not a creative drawing task.



**Foundational Rule: Your absolute top priority is to perfectly preserve the geometry, path, and one-to-one correspondence of every single wire from the original image.**



**1. Canvas and Background:**

* The entire background must be a single, solid color: **black (RGB: 0, 0, 0)**.

* The original image must not be visible in the output.



**2. Absolute Color Uniformity:**

* Every segmented part (an entire wire body or an entire pad) must be filled with a **single, flat, solid color**. No gradients, no shading. Use a digital "paint bucket" fill.



**3. Simplified Coloring Rules (Replaces the old algorithm):**

* **Rule A (Pads):** Identify all connection pads. Color **ALL** of them with one uniform color: **Magenta (RGB: 255, 0, 255)**.

* **Rule B (Wires):** Trace each individual bond wire from one end to the other. As you trace each wire, assign it a color by **sequentially cycling through the following 5-color palette**:

* **Palette: [Red (255,0,0), Green (0,255,0), Blue (0,0,255), White (255,255,255), Yellow (255,255,0)]**

* For example, the first wire you trace is Red, the second is Green, the third is Blue, the fourth is White, the fifth is Yellow, the sixth goes back to Red, and so on. This simple cyclical pattern ensures adjacent wires will likely have different colors without requiring complex analysis from you.



**4. Critical Constraints & Prohibitions (VERY IMPORTANT):**

* **Adhere to Original Paths:** You MUST trace the exact path of each wire as it exists in the source image.

* **Maintain Separation:** DO NOT merge, bundle, or converge different wires into a single point or a tangled mass. Every wire is sacred and must maintain its distinct path.

* **One-to-One Mapping:** There must be a one-to-one correspondence between a wire in the input image and a colored wire in your output. Do not invent new wire paths or fail to trace existing ones.



The final output should look like a human engineer meticulously traced each wire and pad with a specific set of colored pens on a black sheet of paper.

"""

OPTIONAL_HEADERS = {
    # "HTTP-Referer": "https://your-site.example",
    # "X-Title": "YourProjectName"
}

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

MAX_RETRIES = 3
BACKOFF_BASE = 2.0

TIMEOUT_CONNECT = 10
TIMEOUT_READ = 30

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
# ============================================================


def load_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    print("未检测到环境变量 OPENROUTER_API_KEY。")
    key = getpass("请输入 OpenRouter API Key (隐藏输入): ").strip()
    if not key:
        raise ValueError("未提供 API Key。")
    return key


def list_image_files(input_dir: Path) -> List[Path]:
    return [p for p in sorted(input_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def guess_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "image/png"


def load_and_convert_if_needed(path: Path) -> bytes:
    if path.suffix.lower() == ".bmp":
        with Image.open(path) as img:
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGBA")
            buf = BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
    with open(path, "rb") as f:
        return f.read()


def build_data_url(path: Path) -> str:
    if path.suffix.lower() == ".bmp":
        mime = "image/png"
        raw = load_and_convert_if_needed(path)
    else:
        raw = load_and_convert_if_needed(path)
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
                timeout=(TIMEOUT_CONNECT, TIMEOUT_READ)
            )
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            else:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        except KeyboardInterrupt:
            # 直接抛出让上层捕获，优雅退出
            raise
        except Exception as e:
            last_err = e
        attempt += 1
        if attempt < MAX_RETRIES:
            time.sleep(BACKOFF_BASE * attempt)
    raise RuntimeError(f"请求失败（重试 {MAX_RETRIES} 次）: {last_err}")


def extract_base64_images(resp_json: dict) -> List[str]:
    """
    兼容当前多种返回结构：
      - message.images[*].image_url.url (data:image/png;base64,...)
      - message.content[list] 各种 item: image_base64 / b64_json / data / image_url.url
      - message.content 为字符串时包含 data:image
    """
    results: List[str] = []
    for ch in resp_json.get("choices", []):
        msg = ch.get("message", {})
        # images 数组
        images_arr = msg.get("images")
        if isinstance(images_arr, list):
            for item in images_arr:
                if not isinstance(item, dict):
                    continue
                iu = item.get("image_url")
                if isinstance(iu, dict):
                    url = iu.get("url")
                    if isinstance(url, str) and "base64" in url:
                        results.append(url)
        # content 列表
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if "image_base64" in part and isinstance(part["image_base64"], str):
                    results.append(part["image_base64"])
                elif "b64_json" in part and isinstance(part["b64_json"], str):
                    results.append(part["b64_json"])
                elif part.get("type") in ("image", "output_image") and isinstance(part.get("data"), str):
                    results.append(part["data"])
                elif part.get("type") == "image_url":
                    iu2 = part.get("image_url")
                    if isinstance(iu2, dict):
                        url2 = iu2.get("url")
                        if isinstance(url2, str) and "base64" in url2:
                            results.append(url2)
        elif isinstance(content, str):
            if "data:image" in content and "base64" in content:
                results.append(content)
    return results


def save_base64_images(b64_list: List[str], out_dir: Path, base_name: str, timestamp: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for idx, data_str in enumerate(b64_list, start=1):
        original = data_str
        if data_str.startswith("data:"):
            try:
                data_str = data_str.split(",", 1)[1]
            except Exception:
                data_str = original
        try:
            raw = base64.b64decode(data_str)
        except Exception:
            continue
        fname = f"{base_name}_{timestamp}_{idx}{OUTPUT_EXT}"
        (out_dir / fname).write_bytes(raw)
        count += 1
    return count


def process_single_image(img_path: Path, output_root: Path, headers: dict):
    data_url = build_data_url(img_path)
    messages = build_messages(PROMPT, img_path, data_url)
    payload = {
        "model": MODEL,
        "messages": messages,
    }
    resp_json = request_with_retries(payload, headers)
    base64_imgs = extract_base64_images(resp_json)
    ts = time.strftime("%H%M%S", time.localtime())
    if base64_imgs:
        out_subdir = output_root / img_path.stem
        saved = save_base64_images(base64_imgs, out_subdir, img_path.stem, ts)
        return {"status": "ok", "saved": saved}
    else:
        raw_path = output_root / f"{img_path.stem}_raw_{ts}.json"
        raw_path.write_text(json.dumps(resp_json, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"status": "no_image", "saved": 0}


def main():
    print("=== 批量调用 OpenRouter (简化可中断版) ===")
    try:
        api_key = load_api_key()
    except Exception as e:
        print(f"[错误] {e}")
        return

    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        print(f"[错误] 输入目录不存在: {in_dir}")
        return

    images = list_image_files(in_dir)
    if not images:
        print("[提示] 输入目录中没有支持的图片。")
        return

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    headers.update(OPTIONAL_HEADERS)

    print(f"发现 {len(images)} 张图片，开始处理。按 Ctrl+C 可中断（当前请求结束后生效）。\n")

    success = 0
    no_image = 0
    failed = 0

    try:
        for img_path in tqdm(images, desc="Processing", unit="img"):
            try:
                result = process_single_image(img_path, out_dir, headers)
                if result["status"] == "ok":
                    success += 1
                elif result["status"] == "no_image":
                    no_image += 1
            except KeyboardInterrupt:
                print("\n[中断] 捕获 KeyboardInterrupt，停止后续处理。")
                break
            except Exception as e:
                failed += 1
                err_log = out_dir / "error.log"
                with err_log.open("a", encoding="utf-8") as ef:
                    ef.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {img_path.name}: {repr(e)}\n")
    except KeyboardInterrupt:
        print("\n[中断] 用户取消。")

    print("\n=== 结束（可能已被中断） ===")
    print(f"成功保存图片: {success}")
    print(f"未返回图片 : {no_image}")
    print(f"失败       : {failed}")
    print(f"输出目录   : {out_dir.resolve()}")
    if no_image > 0:
        print("提示：若模型不真正生成图片，可换模型或查看 *_raw_*.json 分析结构。")


if __name__ == "__main__":
    main()