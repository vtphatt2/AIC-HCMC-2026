# Synthetic Query Generator

Pipeline tự động sinh multi-modal search queries cho AIC HCMC 2026, dùng Fireworks.ai + Qwen3.7 Plus (Vision).

## Cài đặt

```bash
pip install openai python-dotenv pillow
```

## Cấu hình

```bash
cp .env.example .env
```

Sửa `.env`:

```env
FIREWORKS_API_KEY=fw_xxxxxxxxxxxxxxxxxxxx    # https://fireworks.ai/api-keys
OPENAI_BASE_URL=https://api.fireworks.ai/inference/v1
LLM_MODEL=accounts/fireworks/models/qwen3p7-plus
```

## Chạy

```bash
# Dry-run — kiểm tra extraction, không gọi API
python -m scripts.synthetic_query_generator.pipeline_runner \
  --video L01_V001 --dry-run --limit-scenes 3

# Test 1 scene
python -m scripts.synthetic_query_generator.pipeline_runner \
  --video L01_V001 --limit-scenes 1

# Chạy toàn bộ 9 video (325+ scenes)
python -m scripts.synthetic_query_generator.pipeline_runner
```

Output: `generated_queries/generated_queries_YYYYMMDD_HHMMSS.json`

## Cấu trúc output

```json
{
  "L01_V001": [
    {
      "video_id": "L01_V001",
      "youtube_url": "https://...",
      "fps": 25,
      "scene_start_frame": 0,
      "scene_end_frame": 45,
      "timestamp_range": [0.0, 1.8],
      "keyframe_ids": ["000022.jpg"],
      "generated_queries": [
        {"type": "visual", "query": "Mô tả hành động, đối tượng, bối cảnh..."},
        {"type": "transcript", "query": "Tóm tắt nội dung lời nói..."},
        {"type": "temporal", "query": "Chuỗi sự kiện theo thời gian..."},
        {"type": "natural", "query": "Câu hỏi tự nhiên của người dùng..."}
      ]
    }
  ]
}
```

## Luồng xử lý

```
AIC2026_sample/
  ├── keyframes/           → ảnh .jpg base64
  ├── transnetv2_outputs/  → scene boundaries
  ├── transcripts/         → JSONL lời nói
  └── metadata/            → youtube_url, fps

         ↓ context_extractor.py
    Mỗi scene: 1-3 ảnh + transcript ±10s

         ↓ query_generator.py
    Qwen3.7 Plus (vision) → 4 loại query tiếng Việt

         ↓ pipeline_runner.py
    JSON output vào generated_queries/
```

## Flags

| Flag | Mô tả |
|---|---|
| `--video L01_V001` | Chạy 1 video cụ thể |
| `--dry-run` | Chỉ extract context, không gọi LLM |
| `--limit-scenes N` | Giới hạn số scene để test |

## Đổi provider/model

Sửa `.env`:

```env
# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini

# Gemini (miễn phí)
GEMINI_API_KEY=AIza...
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
LLM_MODEL=gemini-2.0-flash
```
