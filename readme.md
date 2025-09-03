在[openrouter](https://openrouter.ai/)申请api，搜索：
Gemini 2.5 Flash Image Preview (free)

点击api，点击create api key，获得"sk-or-xxx"的api key

![create api key](D:\BaiduSyncdisk\Workspace\use-nano-banana-by-openrouter\assets\create api key.PNG)

## 1. 安装依赖
```bash
pip install requests tqdm pillow python-dotenv
```

## 2. 准备目录
```
input_images/
  sample.bmp
  photo.jpg
```

## 3. 配置 API Key

方式一（推荐，使用 .env）：
```
# 创建 .env
echo OPENROUTER_API_KEY=sk-or-xxxxx > .env   (PowerShell 用: Set-Content .env "OPENROUTER_API_KEY=sk-or-xxxxx")
```

方式二：临时环境变量 (PowerShell)：
```
$env:OPENROUTER_API_KEY="sk-or-xxxxx"
```

方式三：运行时交互输入（如果未设置上面任一方式，会提示隐藏输入）。

## 4. 运行
```bash
python main-win.py
```

## 5. 输出
```
output_results/
  sample/
    sample_HHMMSS_1.png
  photo/
    photo_HHMMSS_1.png
  sample_raw_HHMMSS.json  (若模型未返回生成图)
```

## 6. BMP 处理
- BMP 会在内存中转换为 PNG 上传（data:image/png;base64,...）。
- 其它格式直接读取二进制（如需统一转 PNG，可在 build_data_url 中仿照 BMP 分支改写）。

## 7. 调整返回解析
如果保存的 *_raw_*.json 显示图片 base64 在别的字段，修改 `extract_base64_images` 函数即可。

## 8. 常见问题
| 问题                  | 处理                                |
| --------------------- | ----------------------------------- |
| 401 Unauthorized      | Key 不正确或未设置                  |
| 429 Too Many Requests | 增大 BACKOFF_BASE 或添加 sleep      |
| 没有生成图片          | 模型不支持生成，换模型或检查 JSON   |
| BMP 颜色异常          | 强制使用 img.convert("RGBA") 再保存 |

