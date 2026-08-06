# Preprocess performance refactor plan

Checklist này là nguồn theo dõi tiến độ cho đợt refactor hiệu năng. Mỗi mục chỉ
được đánh dấu hoàn tất sau khi code, test và tài liệu tương ứng đã được cập
nhật. Thiết kế không giả định model GPU cụ thể; CPU/CUDA/MPS và capability được
resolve ở runtime hoặc được yêu cầu rõ bằng config.

## Baseline và nguyên tắc

- [x] Inventory toàn bộ `preprocess/` và xác định dataflow/checkpoint hiện tại.
- [x] Chạy baseline unit test: 59 tests pass.
- [x] Ghi nhận baseline static checks: `compileall` và `pip check` pass; venv
  hiện chưa cài `ruff`.
- [x] Không thay đổi output contract mặc định nếu chưa có migration được docs.
- [x] Không mặc định tên/model GPU; config `auto` phải capability-aware, config
  explicit phải fail-fast.
- [x] Giữ mọi thay đổi trong `preprocess/`; không sửa thay đổi `.gitignore` của
  người dùng.

## 1. Fingerprint, digest và resume

- [x] Thay runtime fingerprint toàn cục bằng fingerprint theo từng stage.
- [x] Fingerprint code phải nhận biết source đang dirty nhưng không invalidate
  stage vì thay đổi docs/module không liên quan.
- [x] Bỏ digest output `scene-segments` khỏi fingerprint của chính stage tạo nó.
- [x] Đưa digest scene segment vào consumer `process_validate`.
- [x] Đưa nội dung Kaggle metadata template vào fingerprint upload.
- [x] Cache SHA-256 theo stat identity để một file lớn chỉ bị đọc một lần trong
  một process.
- [x] Resume dùng tiered validation: stat nhanh trước, content hash khi stat đổi
  hoặc khi chạy full audit.
- [x] Test: pull/code change chỉ invalidate stage liên quan.
- [x] Test: shot-boundary stage ổn định ngay sau lần tạo output đầu tiên.
- [x] Test: checkpoint từng video vẫn resume đúng khi artifact hỏng.

## 2. FFmpeg, render, validate và progress

- [x] Stream `showinfo` từ FFmpeg thay vì capture toàn bộ stderr.
- [x] Thanh `scan` phản ánh frame được decode thực, không replay timeline sau
  khi FFmpeg đã kết thúc.
- [x] Cache timeline theo video/fingerprint để đổi selector không cần scan lại.
- [x] Fast path PNG nguyên kích thước không encode lại bằng Pillow.
- [x] Tránh hash/verify lại artifact vừa được writer tạo và xác nhận.
- [x] Throttle refresh TTY đúng `min_interval_seconds`.
- [x] Không gọi `nvidia-smi` thường hơn `metrics_interval_seconds`.
- [x] Bổ sung `processing.ffmpeg_threads`; `0` để FFmpeg tự resolve portable.
- [x] Test CFR/VFR, mismatch fallback, PNG fast path và progress streaming.

## 3. Embedding device-neutral

- [x] Báo requested/resolved device và capability; explicit CUDA phải fail-fast.
- [x] Không có default phụ thuộc RTX 3060 hoặc VRAM cố định.
- [x] Checkpoint embedding theo từng video.
- [x] Hash mỗi ảnh nguồn tối đa một lần trong một embedding attempt.
- [x] Không `np.load` cùng feature hai lần khi kiểm tra cache.
- [x] Ghi feature/provenance theo batch hoặc video để giảm small-file/fsync cost
  nhưng vẫn đọc được artifact legacy.
- [x] Upload cả feature provenance cần thiết để dataset remote reproducible.
- [x] Test CPU/fake CUDA capability, resume giữa video và cache invalidation do
  model/precision/autocast/TF32.

## 4. Archive, staging, upload và cleanup

- [x] Không CRC/hash ZIP nhiều lượt trước extraction; extraction tạm vẫn phải
  phát hiện CRC lỗi trước atomic publish.
- [x] Staging dùng hardlink khi cùng filesystem và fallback copy khác filesystem.
- [x] Payload digest được tính một lần và được tái sử dụng trong cùng upload.
- [x] Cleanup ngay sau upload không reverify/hash lại payload vừa verify; lệnh
  cleanup độc lập vẫn reverify remote.
- [x] `mode=auto` xử lý rõ Kaggle CLI trả 403 cho dataset chưa tồn tại.
- [x] Upload progress không bị giữ toàn bộ trong RAM.
- [x] Remote verification phân biệt `ready` với content verification và lưu
  evidence đủ để cleanup an toàn.
- [x] Disk guard extraction dùng uncompressed archive; upload ghi logical payload,
  temporary reserve và free disk vào `upload-disk-budget.json`.
- [x] Test create/version/403, interrupted upload, cleanup guard và lot staging.

## 5. Scheduling và concurrency

- [x] Cho phép upload lot N overlap xử lý lot N+1 khi bật config.
- [x] Dataset-scoped/cumulative upload vẫn serialize bằng dataset lock.
- [x] Giới hạn một upload pending và kiểm tra temporary disk reserve.
- [x] Progress nền tắt; state/upload-state và lock không ghi đè nhau.
- [x] Test overlap bounded, lock, interruption, continue-on-error và output order.

## 6. Docs và xác nhận cuối

- [x] Cập nhật `config.example.json` cho mọi option mới và migration default.
- [x] Cập nhật README dataflow, performance profiles, resume và tmux workflow.
- [x] Docs rõ filename `dataset-metadata.json` nguồn và file staging.
- [x] Docs rõ transcript layout theo lot và PNG/JPEG extension.
- [x] Chạy unit tests, compileall, pip check và lint/static check: 72 tests,
  `compileall`, `pip check`, `ruff check`, config parse, CLI help và
  `git diff --check` đều pass.
- [x] Chạy benchmark nhẹ cho hash cache, FFmpeg timeline/render và embedding
  fake encoder; ghi kết quả tương đối, không khẳng định theo GPU cụ thể.
- [x] Review `git diff`: toàn bộ thay đổi của đợt refactor nằm trong
  `preprocess/`; `.gitignore` đang dirty từ trước và không bị đợt này chỉnh sửa.
- [x] Giữ push guard: hiện chưa push; nếu được yêu cầu thì chỉ push branch
  `preprocess`, không merge `main`.

## Follow-up không nằm trong thay đổi bắt buộc này

- [ ] Giữ cùng DataLoader worker process qua ranh giới video. Thiết kế hiện tại
  ưu tiên checkpoint/restore atomically theo video; persistent worker của
  PyTorch không thể đổi dataset an toàn sau khi worker đã fork nếu không đổi
  sang một lot-wide sampler.
- [ ] Overlap render CPU và embedding GPU trong cùng lot. Việc này cần state
  writer/progress đa luồng và bounded artifact queue riêng; không bật vội vì có
  thể làm hỏng checkpoint đang là guarantee chính của SSH resume.
- [ ] Download lại `provenance.json` từ Kaggle để byte-compare remote. CLI
  `datasets files` hiện chỉ cho status/inventory, không trả checksum nội dung.

## Kết quả benchmark nhẹ

- SHA-256 file PNG 539,949 bytes, cache hit: `18.4 us/call`; đọc/hash lại:
  `368 us/call` (khoảng 20x trong benchmark local; video lớn sẽ phụ thuộc disk).
- FFmpeg timeline cache, PNG fast path và embedding fake encoder được đo bằng
  regression suite. Smoke test FFmpeg CFR 160x90/25 fps render được 4 PNG đúng
  kích thước gốc; không ghi con số GPU vì test không dùng model/device thật.
