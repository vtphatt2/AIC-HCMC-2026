import json
import time
import logging
from . import config

logger = logging.getLogger(__name__)


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


SYSTEM_PROMPT = """Bạn là một chuyên gia dữ liệu đa phương thức (Multi-modal Data Specialist) cho một cuộc thi truy xuất video (Video Retrieval Challenge).

NHIỆM VỤ: Dựa vào ảnh keyframe và transcript của một scene video, bạn tạo ra các câu query tìm kiếm mà người dùng có thể nhập vào hệ thống.

YÊU CẦU:
1. Viết query bằng tiếng Việt, tự nhiên như người dùng thật.
2. Sinh từ 3-4 query đa dạng, mỗi query thuộc một loại:
   - "visual": Mô tả hành động, màu sắc, đối tượng, bối cảnh hiển thị trong ảnh.
   - "transcript": Tóm tắt/trích ý nội dung lời nói trong transcript (nếu có).
   - "temporal": Mô tả chuỗi sự kiện diễn ra trong scene.
   - "natural": Câu hỏi tự nhiên như người dùng chỉ nhớ thoáng qua.
3. Query phải ngắn gọn (1-3 câu), cụ thể, dễ hiểu.
4. Trả về JSON với cấu trúc chính xác như trong ví dụ.

Định dạng JSON bắt buộc:
{
  "generated_queries": [
    {"type": "visual", "query": "..."},
    {"type": "transcript", "query": "..."},
    {"type": "temporal", "query": "..."},
    {"type": "natural", "query": "..."}
  ]
}

Nếu transcript trống thì bỏ qua loại "transcript".
Chỉ trả về JSON, không thêm bất kỳ văn bản nào khác."""


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
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "low"}
        })

    for attempt in range(config.MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=1000,
            )
            result = json.loads(response.choices[0].message.content)
            return result.get("generated_queries", [])
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < config.MAX_RETRIES - 1:
                time.sleep(config.BACKOFF_SEC * (attempt + 1))

    logger.error(f"All {config.MAX_RETRIES} attempts failed for {context['video_id']} scene {context['scene_start_frame']}")
    return []
