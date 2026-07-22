import json
import re
import time
import logging
from . import config

logger = logging.getLogger(__name__)


def _extract_json(content: str) -> dict | None:
    """Extract JSON from content that may contain thinking prefix text."""
    if not content:
        raise ValueError("Empty response from model")

    # Try direct parse first
    stripped = content.strip()
    if stripped.startswith("{"):
        return json.loads(stripped)

    # Look for JSON in <json>...</json> or ```json...``` blocks
    for pattern in [r"<json>\s*([\s\S]*?)\s*</json>", r"```json\s*([\s\S]*?)\s*```"]:
        m = re.search(pattern, content)
        if m:
            return json.loads(m.group(1))

    # Fallback: find first { and try to parse from there
    idx = content.find("{")
    if idx >= 0:
        return json.loads(content[idx:])

    raise ValueError(f"No JSON found in response. First 200 chars: {content[:200]}")


def _try_create(client, model: str, messages: list) -> dict | None:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=3000,
        )
        content = response.choices[0].message.content
        return _extract_json(content)
    except Exception as e:
        raise e


def _load_fewshot_examples() -> str:
    """Read all query samples from queries_1/ and queries_2/ as few-shot examples."""
    examples = []
    for queries_root in [config.QUERIES_DIR, config.QUERIES_DIR_1]:
        if not queries_root.exists():
            continue
        for f in queries_root.rglob("*.txt"):
            content = f.read_text().strip()
            if content:
                examples.append(content)
    if not examples:
        return ""
    # pick up to 15 diverse examples
    import random
    random.seed(42)
    selected = random.sample(examples, min(15, len(examples)))
    return "\n\n---\n\n".join(f"Ví dụ {i+1}:\n{e}" for i, e in enumerate(selected))


SYSTEM_PROMPT = """Bạn là chuyên gia sinh query tìm kiếm video bằng tiếng Việt.

NHIỆM VỤ: Dựa vào ảnh keyframe và transcript của một scene, tạo 3-4 câu query.

LUẬT:
1. Query tiếng Việt, tự nhiên, ngắn gọn (1-2 câu).
2. Mỗi query thuộc một loại:
   - "visual": mô tả hành động, đối tượng, màu sắc, bối cảnh trong ảnh.
   - "transcript": tóm tắt/trích ý lời nói (bỏ qua nếu transcript trống).
   - "temporal": mô tả chuỗi sự kiện theo thời gian.
   - "natural": câu hỏi tự nhiên như người dùng chỉ nhớ lờ mờ.
3. Kết thúc câu trả lời bằng JSON trong cặp thẻ <json></json>.
4. Viết <json> ở dòng riêng, KHÔNG có text nào khác sau </json>.

Định dạng bắt buộc:
<json>
{
  "generated_queries": [
    {"type": "visual", "query": "mô tả..."},
    {"type": "transcript", "query": "..."},
    {"type": "temporal", "query": "..."},
    {"type": "natural", "query": "..."}
  ]
}
</json>"""


def _build_user_message(context: dict, fewshot: str) -> str:
    video_id = context["video_id"]
    ts = context["timestamp_range"]
    transcript = context.get("transcript_text", "")

    msg = f"""Video: {video_id}
Thời gian scene: {ts[0]}s → {ts[1]}s (frame {context['scene_start_frame']} → {context['scene_end_frame']})
Transcript: {transcript if transcript else "(không có lời thoại trong scene này)"}

Ảnh keyframe được đính kèm bên dưới. Hãy tạo các query tìm kiếm cho scene này."""

    if fewshot:
        msg += f"\n\nDưới đây là các ví dụ query từ các cuộc thi trước để tham khảo phong cách:\n\n{fewshot}"

    return msg


def generate_queries(context: dict, client) -> list[dict]:
    fewshot = _load_fewshot_examples()

    content = [{"type": "text", "text": _build_user_message(context, fewshot)}]
    for img_b64 in context.get("images_base64", []):
        if len(img_b64) > config.MAX_IMAGE_SIZE_BYTES * 2:
            logger.warning(
                f"Base64 image {len(img_b64)} bytes exceeds limit, skipping. "
                f"scene={context['scene_start_frame']}"
            )
            continue
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "low"}
        })

    if len(content) == 1:
        logger.warning(f"No images for scene {context['scene_start_frame']}, text-only prompt")
        content[0]["text"] += "\n\n(Lưu ý: không có ảnh keyframe nào khả dụng, hãy dựa vào transcript và thời gian để sinh query.)"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]

    models_to_try = [config.OPENAI_MODEL] + [
        m for m in config.MODEL_FALLBACKS if m != config.OPENAI_MODEL
    ]

    for model in models_to_try:
        for attempt in range(config.MAX_RETRIES):
            try:
                result = _try_create(client, model, messages)
                if model != config.OPENAI_MODEL:
                    logger.info(f"Model fallback: {config.OPENAI_MODEL} → {model} (worked)")
                return result.get("generated_queries", [])
            except Exception as e:
                err = str(e)
                if "API key" in err or "INVALID_ARGUMENT" in err or "auth" in err.lower():
                    logger.error(f"Invalid API key — check GEMINI_API_KEY in .env. Got: {err[:120]}")
                    return []
                if "429" in err or "quota" in err.lower():
                    wait = config.BACKOFF_SEC * (2 ** attempt)
                    logger.warning(f"Rate limited (429). Waiting {wait:.0f}s...")
                    time.sleep(wait)
                    continue
                if "404" in err or "not found" in err.lower():
                    logger.warning(f"Model '{model}' not found (404), trying next...")
                    break
                logger.warning(f"Attempt {attempt + 1} with {model} failed: {err[:120]}")
                if attempt < config.MAX_RETRIES - 1:
                    time.sleep(config.BACKOFF_SEC * (attempt + 1))
        else:
            continue

    logger.error(f"All models exhausted for {context['video_id']} scene {context['scene_start_frame']}")
    return []
