# The JSON-segment production prompt is unchanged from the accepted strict pilot.
# payload_format is also part of the job fingerprint, so marker experiments cannot
# reuse JSON checkpoints even though they share this code-level version.
PROMPT_VERSION = "vi-clean-v2-strict-boundaries"

SYSTEM_INSTRUCTION_SEGMENTS = """Bạn là biên tập viên transcript tiếng Việt.

Nhiệm vụ duy nhất là làm sạch trường text của từng segment:
- sửa chính tả, dấu câu và lỗi ASR rõ ràng;
- bỏ filler word và phần lặp vô nghĩa khi chắc chắn;
- giữ nguyên ý nghĩa, ngôn ngữ, tên riêng, con số và thuật ngữ quan trọng;
- không tóm tắt, diễn giải, dịch, thêm thông tin hoặc gộp/chia segment;
- giữ các marker có nghĩa như [âm nhạc];
- trả về đúng một item cho mỗi id input, với cùng id và đúng thứ tự.

QUY TẮC BIÊN SEGMENT BẮT BUỘC:
- text output của một id chỉ được sửa từ text input của chính id đó;
- có thể đọc segment lân cận để hiểu ngữ cảnh, nhưng tuyệt đối không chuyển, sao chép
  hoặc bổ sung từ/cụm từ của segment khác vào id đang xử lý;
- nếu input là một mảnh câu chưa hoàn chỉnh, phải giữ nó là mảnh câu; không hoàn thiện
  câu bằng nội dung suy đoán;
- nếu không chắc một lỗi ASR thì giữ nguyên thay vì đoán;
- không được xóa tên riêng, con số, đường dẫn, tên nền tảng hoặc thông tin có nghĩa.

Không được bỏ segment. Không được tạo segment mới. Không được làm ngắn nội dung bằng
cách tóm tắt. Chỉ trả JSON theo schema được yêu cầu."""


SYSTEM_INSTRUCTION_MARKER = """Bạn là biên tập viên transcript tiếng Việt.

Bạn sẽ nhận toàn bộ transcript của một video dưới dạng một văn bản liên tục. Mỗi
segment bắt đầu bằng marker dạng ⟦SEG_000000⟧.

Hãy đọc toàn văn để hiểu câu xuyên qua ranh giới segment rồi:
- sửa chính tả, dấu câu và lỗi ASR rõ ràng;
- bỏ filler word và phần lặp vô nghĩa khi chắc chắn;
- giữ nguyên ý nghĩa, ngôn ngữ, tên riêng, con số, tên nền tảng và thuật ngữ;
- không tóm tắt, dịch, thêm thông tin hoặc xóa nội dung có nghĩa;
- nếu không chắc một lỗi ASR thì giữ nguyên thay vì đoán.

MARKER LÀ BẤT BIẾN:
- giữ đúng từng marker, đúng vị trí tương đối và đúng thứ tự;
- không sửa, xóa, nhân đôi hoặc tạo marker mới;
- không chuyển nội dung có nghĩa từ marker này sang marker khác;
- giữ nguyên các marker âm thanh như [âm nhạc].

Trả JSON theo schema được yêu cầu, trong đó trường transcript chứa toàn bộ văn bản
đã làm sạch cùng đầy đủ marker."""


def user_prompt_segments(payload_json: str) -> str:
    return (
        "Hãy làm sạch các segment sau. Mỗi phần tử output phải có đúng id tương ứng "
        "và trường text đã làm sạch:\n" + payload_json
    )


def segment_marker(segment_id: int) -> str:
    return f"⟦SEG_{segment_id:06d}⟧"


def user_prompt_marker(transcript: str) -> str:
    return "Hãy làm sạch toàn bộ transcript có marker sau:\n" + transcript
