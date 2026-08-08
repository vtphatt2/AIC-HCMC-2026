# Preprocess

Các công cụ trong thư mục này xử lý transcript, keyframe và feature visual của
dataset. Chúng là pipeline offline, không cần chạy FastAPI, và dùng virtual
environment riêng tại `preprocess/.venv`. Toàn bộ workflow SSH chỉ chạm vào
`preprocess/` và `data/`; không phụ thuộc vào `app/`.

Nếu cần chuyển công việc sang một coding/SSH session khác, bắt đầu tại
[`SESSION_HANDOFF.md`](SESSION_HANDOFF.md). Tài liệu đó tổng hợp trạng thái hiện
tại của kiến trúc, contract stage, checkpoint/resume, config và runbook; README
này giữ phần tham chiếu chi tiết cho từng pipeline.

## Cấu trúc source

```text
preprocess/
├── README.md
├── SESSION_HANDOFF.md          # context đầy đủ để tiếp tục ở session khác
├── pecore/                     # PE-Core image encoder → normalized .npy
├── batch/                      # download → validate → process → embed → Kaggle
└── transcript/
    ├── sentences.py             # fragment TXT → sentence có interval thời gian
    ├── frame_assignment.py      # chọn một anchor keyframe cho một sentence
    ├── build_keyframe_index.py  # CLI: transcript TXT → JSON sentence index
    └── add_transcript_banner.py # CLI: JSON index + anchor JPG → ảnh có banner
```

## Điều kiện chạy

Chạy command từ root repository (`AIC-HCMC-2026/`) sau khi kích hoạt venv:

```bash
source preprocess/.venv/bin/activate
python <script> [arguments]
```

Pipeline lock này hỗ trợ Python **3.10 trở lên**. Với Python 3.10, pip sẽ
chọn `kaggle==1.7.4.5`; từ Python 3.11 trở lên sẽ chọn `kaggle==2.2.4`.
Không dùng Python 3.9 hoặc thấp hơn vì các package vision trong lock có thể
không có wheel tương thích.

Xem chính xác argument của một command:

```bash
preprocess/.venv/bin/python -m \
  preprocess.transcript.build_keyframe_index --help
```

## Dataset layout và quy ước ID

Các script transcript hỗ trợ cả layout sample cũ và layout SSH dưới `data/`.
`build_keyframe_index.py` mặc định dùng `<repository-root>/data`; banner
renderer giữ mặc định sample cũ để tương thích và nhận `--data-root` alias.
Layout tương ứng là:

```text
AIC2026_sample/
├── transcripts/<video_id>_Transcript.txt
├── metadata/<video_id>.json       # fps có thể thiếu
├── PECore-features/<video_id>/<frame_id>.npy
├── keyframes/<video_id>/<frame_id>.jpg
│   # local fallback supported: keyframes_org/<video_id>/<frame_id>.jpg
├── keyframe_transcript_index/<video_id>.json    # generated
└── subtitled_keyframes/<video_id>/<frame_id>.jpg # generated
```

- `video_id` là ID thư mục, ví dụ `L01_V001`.
- `frame_id` là sáu chữ số cuối của tên ảnh, ví dụ `000237`. Tên file có thể
  là `000237.jpg` hoặc `L01_V001_000237.jpg`; renderer lấy nhóm số cuối cùng.
  Không ghép `video_id` vào `frame_id` trong index.
- Nếu cả hai tên file đại diện cùng một `frame_id` cùng tồn tại, renderer chỉ
  xử lý một ảnh và ưu tiên tên chuẩn `000237.jpg`.
- Nếu rendered manifest có `timestamp_ms`/PTS, đó là timestamp có thẩm quyền;
  chỉ fallback sang `int(int(frame_id) / fps * 1000)` khi metadata có FPS.
- File transcript có một fragment trên mỗi dòng:

  ```text
  [HH:MM:SS] Nội dung transcript
  [HH:MM:SS.sss] Nội dung transcript có milliseconds
  ```

  Dòng không khớp format bị bỏ qua; tiền tố `>>` được loại khỏi text.

## Pipeline 1: transcript → keyframe transcript index

Script: `preprocess/transcript/build_keyframe_index.py`

```text
transcripts/<video_id>_Transcript.txt
  → fragment có timestamp
  → sentence không overlap theo dấu câu
  → anchor keyframe gần midpoint
  → keyframe_transcript_index/<video_id>.json
```

Script đọc trực tiếp TXT; **không đọc và không ghi**
`AIC2026_sample/transcripts_processed/`.

### Input bắt buộc

| Đường dẫn | Vai trò |
| --- | --- |
| `transcripts/<video_id>_Transcript.txt` | Nguồn transcript. |
| `metadata/<video_id>.json` | Lấy `fps` nếu có; thiếu vẫn hợp lệ khi rendered manifest có PTS. |
| `PECore-features/<video_id>/*.npy` | Danh sách keyframe ưu tiên để chọn anchor. |
| `keyframes/<video_id>/*.jpg` | Chỉ dùng làm fallback nếu không có `.npy` cho video đó. |

### CLI arguments

| Argument | Bắt buộc | Mặc định | Hành vi |
| --- | --- | --- | --- |
| `--video-id <id>` | Không | Build tất cả transcript TXT | Chỉ build video chỉ định. Có thể lặp: `--video-id L01_V001 --video-id L01_V002`. |
| `--force` | Không | Tắt | Ghi lại output dù JSON đã mới hơn TXT nguồn. Không có cờ này, video hiện tại bị `SKIP`. |

### Cách chạy

Build toàn bộ:

```bash
preprocess/.venv/bin/python -m \
  preprocess.transcript.build_keyframe_index --force
```

Build một video:

```bash
preprocess/.venv/bin/python -m \
  preprocess.transcript.build_keyframe_index \
  --video-id L03_V002 --force
```

### Quy tắc tạo sentence và anchor

1. Các fragment được nối bằng khoảng trắng, sau đó tách tại `.`, `!`, `?`,
   hoặc `…`.
2. Nếu một đoạn thiếu dấu kết câu và vượt 15 giây hoặc 320 ký tự, fallback sẽ
   cắt tại boundary fragment gốc. Vì vậy một transcript ASR không có punctuation
   không thể trở thành một sentence duy nhất bao phủ cả video. Khoảng thời gian
   rỗng giữa các fragment được giữ là không có transcript.
3. Mỗi sentence có interval nửa mở `[start_ms, end_ms)`. Sentence kề nhau dùng
   chung boundary nên không overlap thời gian hoặc phần text nguồn.
4. Một anchor phải có timestamp nằm trong interval sentence.
5. Trong các frame hợp lệ, chọn frame gần midpoint của sentence nhất.
6. Nếu không có frame hợp lệ, đặt `anchor_frame_id: null`; không gán nhầm
   frame nằm ngoài câu.

### Output JSON

Output nằm tại `keyframe_transcript_index/<video_id>.json`. Đây là index theo
sentence, nên text không bị lặp lại cho mọi keyframe:

```json
{
  "video_id": "L01_V001",
  "fps": null,
  "timing_source": "rendered_manifest_pts",
  "keyframe_count": 663,
  "sentence_count": 175,
  "anchored_sentence_count": 172,
  "sentences": [
    {
      "sentence_id": "S000000",
      "start_ms": 8000,
      "end_ms": 13462,
      "text": "Kính chào và cảm ơn quý vị...",
      "anchor_frame_id": "000311"
    }
  ]
}
```

Lưu ý: vì index ưu tiên `.npy`, nó bảo đảm feature tồn tại. Việc JPG thật có
tồn tại được banner renderer kiểm tra trước khi render.

## Pipeline 2: render banner transcript

Script: `preprocess/transcript/add_transcript_banner.py`

Script **không** đọc TXT transcript và không dùng transcript window. Nó đọc
`fps` cùng các interval `sentences[]` từ index, quét **toàn bộ** keyframe của
video và map frame theo timestamp:

```text
keyframe_transcript_index/<video_id>.json
+ keyframes/<video_id>/<frame_id>.jpg
→ subtitled_keyframes/<video_id>/<frame_id>.jpg
```

Với mỗi frame, script đọc PTS `timestamp_ms` từ
`keyframes/<video_id>/manifest.json`; nếu PTS không có thì dùng
`int(int(frame_id) / fps * 1000)`. Nếu timestamp nằm trong
`[start_ms, end_ms)` của một sentence, panel chứa `sentence.text`; nếu không,
script vẫn tạo ảnh với panel trống. `anchor_frame_id` không được dùng bởi
renderer này. Với frame có transcript, panel thêm prefix
`Audio content (in Vietnamese): ` trước text sentence.

### Input bắt buộc

| Đường dẫn | Vai trò |
| --- | --- |
| `keyframe_transcript_index/<video_id>.json` | Nguồn `fps`, interval và text sentence. |
| `keyframes/<video_id>/<frame_id>.{jpg,png}` | Toàn bộ ảnh source để render. Nếu không có, renderer fallback sang `keyframes_org/<video_id>/`. Batch SSH mặc định PNG. |

Nếu thư mục keyframe không tồn tại hoặc file ảnh có stem không phải số, script
dừng với lỗi. Không có `continue_on_error`, nhằm phát hiện dữ liệu keyframe bị
lệch ngay lập tức.

### CLI arguments

| Argument | Bắt buộc | Mặc định | Hành vi |
| --- | --- | --- | --- |
| `--video-id <id>` | Có | — | Video cần render. |
| `--sample-root <path>` | Không | `<repo>/AIC2026_sample` | Root dataset thay thế. |
| `--data-root <path>` | Không | — | Alias của `--sample-root`, tiện dùng với `data/`. |
| `--output-dir <path>` | Không | `<sample-root>/subtitled_keyframes` | Root output. Script tạo thêm thư mục `<video_id>/`. |
| `--image-extension <ext>` | Không | `.jpg` | Extension ảnh nguồn. Với keyframe từ batch SSH, truyền `.png`. |
| `--font-path <path>` | Không | Tự tìm DejaVu/Noto/Liberation | Font Unicode `.ttf`/`.otf`. |
| `--font-size <n>` | Không | `30` | Cỡ font pixel; phải dương. |
| `--panel-height <n>` | Không | `200` | Chiều cao panel tối thiểu, pixel; phải dương. |
| `--overflow <mode>` | Không | `error` | `expand`: tăng panel; `truncate`: cắt dòng dư; `error`: dừng nếu panel quá thấp. |
| `--existing <mode>` | Không | `overwrite` | `overwrite`: ghi đè; `skip`: bỏ ảnh đã có; `unique`: thêm `_1`, `_2`, … vào tên file. |
| `--dry-run` | Không | Tắt | Kiểm tra index và danh sách keyframe, không ghi ảnh. |

### Cách chạy

Kiểm tra input không ghi output:

```bash
preprocess/.venv/bin/python -m \
  preprocess.transcript.add_transcript_banner \
  --video-id L03_V002 --dry-run
```

Render với panel tự mở rộng:

```bash
preprocess/.venv/bin/python -m \
  preprocess.transcript.add_transcript_banner \
  --video-id L03_V002 \
  --panel-height 200 \
  --overflow expand
```

Render vào vị trí khác, không ghi đè ảnh cũ:

```bash
preprocess/.venv/bin/python -m \
  preprocess.transcript.add_transcript_banner \
  --video-id L03_V002 \
  --output-dir /tmp/aic-banners \
  --existing unique
```

## Pipeline 3: MP4 → selected keyframes

Script: `preprocess/extract_keyframes.py`. Pipeline tách việc chọn frame khỏi
việc render ảnh: manifest selection giữ source frame number và PTS timestamp,
còn render profile chỉ quyết định kích thước/encoding của ảnh. Vì vậy có thể
render lại từ selection đã kiểm chứng mà không thay đổi thuật toán chọn frame.

Input/output đều là argument bắt buộc; script không giả định tên hay vị trí thư
mục video. Selector mặc định là uniform theo thời gian, mặc định **1 giây**.
Profile mặc định giữ nguyên kích thước frame sau khi decode, không resize và ghi
**PNG lossless** (compression level 6). Nếu muốn giảm kích thước, truyền
`--short-edge <pixels>` hoặc đặt `processing.target_short_edge_px` trong config.

```bash
preprocess/.venv/bin/python -m preprocess.extract_keyframes \
  --input '/absolute/path/to/VIDEO_001.mp4' \
  --video-id VIDEO_001 \
  --output-root /absolute/path/to/keyframe-output
```

Output có hai artifact độc lập:

```text
<output-root>/
├── selection-manifests/VIDEO_001.json
└── original-png/VIDEO_001/
    ├── 000000.png
    └── manifest.json
```

### Selector scene segments (TransNetV2)

`docs/keyframe_selection.md` mô tả selector rule-based dựa trên scene segment
của TransNetV2. Dùng `--selector scene-segments` cùng JSON detector-agnostic:

```json
{
  "segments": [
    {"start_ms": 0, "end_ms": 2500},
    {"start_ms": 2500, "end_ms": 12000}
  ]
}
```

Các segment phải không overlap. Selector chọn frame gần nhất với tâm của các
bucket thời lượng: scene `<=3s` lấy 1 frame, `<=10s` lấy 3 frame, và dài hơn
lấy 5 frame. Ví dụ scene 25 giây có target 10/30/50/70/90% duration.

```bash
preprocess/.venv/bin/python -m preprocess.extract_keyframes \
  --input '/absolute/path/to/VIDEO_001.mp4' \
  --video-id VIDEO_001 \
  --output-root /absolute/path/to/keyframe-output \
  --selector scene-segments \
  --scene-segments /absolute/path/to/transnetv2-scenes.json
```

Để xuất JPEG thay vì PNG, thêm `--image-format jpeg --jpeg-quality 92`.

### Selector `linear-rulebase`

Đây là strategy lấy số frame theo độ dài của từng shot/scene. Nó dùng cùng
scene manifest với `scene-segments`, nhưng số frame do `processing.linear_rule`
quyết định; các mốc không nằm trong code selector:

```json
{
  "processing": {
    "selector": "linear-rulebase",
    "scene_segments_dir": "../../data/scene-segments",
    "linear_rule": {
      "short_duration_ms": 1000,
      "short_frame_count": 1,
      "base_duration_ms": 3000,
      "base_frame_count": 2,
      "increment_duration_ms": 3000,
      "increment_frame_count": 1
    }
  }
}
```

Với config mặc định: shot `<=1s` lấy 1 frame; shot `>1s` và `<=3s` lấy 2
frame; shot dài hơn 3 giây tăng `increment_frame_count` sau mỗi
`increment_duration_ms`. Frame được đặt gần tâm các bucket thời lượng. Ví dụ
1s → 1 frame, 3s → 2 frame, 6s → 3 frame, 9s → 4 frame. Strategy nhận cả
manifest dạng `start_ms/end_ms` và JSON trực tiếp do `transnetv2-pytorch` xuất
với `start_time/end_time`.

Khi `selector` là `linear-rulebase` hoặc `scene-segments`, batch pipeline tự
chạy stage TransNetV2 trước khi chọn keyframe. Nếu không khai báo đường dẫn,
stage dùng mặc định `data_root/scene-segments`; mỗi video được ghi thành
`<video_id>.json`. Nếu manifest đã có và `shot_boundary.overwrite=false`, stage
chỉ validate rồi dùng lại file đó. Muốn dùng manifest tạo sẵn, đặt
`shot_boundary.enabled=false` và khai báo `processing.scene_segments_dir`.

Kiểm tra media và selection mà không ghi gì:

```bash
preprocess/.venv/bin/python -m preprocess.extract_keyframes \
  --input '/absolute/path/to/VIDEO_001.mp4' \
  --video-id VIDEO_001 \
  --output-root /tmp/unused-keyframe-output \
  --dry-run
```

`preprocess/inspect_video.py` đọc bằng cùng backend và chỉ in codec, duration,
FPS, dimensions và frame count. Pipeline cần `ffmpeg`/`ffprobe` trên `PATH`;
timestamp được lấy từ PTS decoder thay vì tự tính bằng FPS.

## Dependencies

Tất cả command trong thư mục này dùng environment riêng của `preprocess`:

```bash
source preprocess/.venv/bin/activate
python -m pip install -r preprocess/requirements.lock.txt
```

Kiểm tra Python và version Kaggle sau khi cài:

```bash
python --version
python -m pip show kaggle | grep '^Version:'
python -m pip check
```

## Hướng dẫn mở rộng code

- Thêm preprocessing transcript trong `preprocess/transcript/`; không thêm vào
  `app/services/transcript_index.py` trừ khi chức năng đó cần phục vụ runtime
  API.
- Giữ `frame_id` là chuỗi sáu chữ số. Video được xác định bằng `video_id` cấp
  JSON và thư mục keyframe.
- Không nhân bản `text` trên từng frame. Nếu cần reverse lookup, tính timestamp
  từ `frame_id / fps` rồi tìm sentence theo `[start_ms, end_ms)` hoặc chỉ lưu
  `sentence_id`.
- Không gán anchor ngoài interval sentence; dùng `null` nếu không có keyframe
  phù hợp.
- Khi thêm hoặc đổi CLI, cập nhật bảng argument, ví dụ command, input/output
  và failure mode trong README này.
- Output dưới `AIC2026_sample/` là dữ liệu tạo sinh; không sửa tay. Chạy lại
  script với `--force` sau khi thay đổi input hoặc thuật toán.

## Pipeline 4: SSH batch download → validate → process → Kaggle

Các module OOP cho workflow SSH nằm trong `preprocess/batch/`. Chúng không
đọc path tuyệt đối cố định; downloader, archive extractor, selector, processor,
validator, staging và uploader đều có boundary để thay implementation khác.

### Cấu trúc chạy tối giản

```text
repository-root/
├── preprocess/
│   ├── plan.md                  # checklist/refactor progress + verification
│   └── batch/
│       ├── cli.py
│       ├── config.example.json
│       ├── embedding.py
│       ├── downloader.py
│       ├── processor.py
│       ├── validators.py
│       ├── kaggle_uploader.py
│       └── cleanup.py
└── data/
    ├── input/links.txt
    ├── metadata/<video_id>.json
    ├── scene-segments/<video_id>.json
    └── <lot_id>/
        ├── archive/
        ├── source/<lot_id>/
        ├── dataset/
        ├── kaggle-staging/            # snapshot riêng lot; xóa sau upload verify
        ├── reports/
        │   ├── decoder-timelines/      # cache frame n + PTS theo source fingerprint
        │   └── embedding-completions/ # journal atomic từng video khi overlap
        ├── receipts/
        ├── upload-state.json
        ├── upload.lock
        └── state.json

# Chỉ xuất hiện khi bật staging_scope=dataset (legacy cumulative mode):
data/kaggle-dataset-staging/
data/kaggle-dataset-state.json
data/kaggle-dataset-upload.lock
```

`data/<lot_id>/` được tạo khi chạy. `source/` chứa video tạm với đúng tên gốc,
ví dụ `L21_V030.mp4`. Tên thư mục `video` ở trong ZIP chỉ được đổi thành tên
lot:

```text
Videos_L29_a.zip → unzip → video/ → data/L29_a/source/L29_a/
```

`video_id` vẫn lấy từ tên file gốc (`L21_V030`), không lấy từ tên lot và không
đổi tên file video.

### Input

`data/input/links.txt` có một URL ZIP trên mỗi dòng; dòng trống và dòng bắt đầu
bằng `#` được bỏ qua:

```text
https://aic-data.ledo.io.vn/Videos_L29_a.zip
https://aic-data.ledo.io.vn/Videos_L30_b.zip
```

`LinkListParser` lấy tên archive từ URL, bỏ query string khi đặt tên file,
đồng thời suy ra `lot_id` bằng cách bỏ prefix cấu hình `Videos_`. Metadata nằm
ngoài archive, độc lập với video:

```text
data/metadata/L21_V030.json
```

Downloader mặc định tương đương `aria2c -x 16 -s 16`:
`max_concurrent_connections` điều khiển số connection tối đa tới một server,
còn `split_count` điều khiển số segment của mỗi archive. Pipeline vẫn thêm
resume, retry, timeout, tên file cố định và validate ZIP sau khi tải.

FPS trong metadata là tùy chọn. Pipeline đo duration, dimensions, codec,
frame count và FPS nếu media probe cung cấp; việc chọn frame dùng PTS thực tế,
do đó video VFR hoặc video không có FPS không bị suy diễn FPS giả.

### Dataflow và I/O contract của từng stage

```text
links.txt + metadata + config
        ↓
preflight
        ↓
aria2c download
        ↓
ZIP validate
        ↓
unzip video/ → source/<lot_id>/
        ↓
video discovery
        ↓
TransNetV2 shot-boundary detection (khi selector cần scene)
        ↓
selection strategy
        ↓
FFmpeg render keyframes
        ↓
video/artifact validation
        ↓
PE-Core image embedding (khi bật)
        ↓
Kaggle staging (không có video)
        ↓
Kaggle upload + status verify
        ↓
cleanup artifact tạm
```

| Stage | Input | Output | Kiểu/chiều và đơn vị quan trọng |
| --- | --- | --- | --- |
| `LinkListParser` | `data/input/links.txt` | `ArchiveInput[]` | Mỗi dòng là `str`; output gồm `url`, `archive_name`, `lot_id`, `line_number`. |
| `PreflightChecker` | `BatchConfig` | `PreflightResult` | `tools: dict[str, str]`, `free_bytes: int`; kiểm tra `aria2c`, `ffmpeg`, `ffprobe`, TransNetV2 khi selector cần scene và `kaggle` nếu upload bật. |
| `Aria2ArchiveDownloader` | `ArchiveInput` + thư mục đích | ZIP + `DownloadResult` | `--split 16`, `--max-connection-per-server 16`; retry/timeout tính bằng giây; dung lượng file tính bằng bytes. |
| `ZipArchiveValidator` | `Path` tới `.zip` | `ArchiveInspection` | `members: tuple[str, ...]`, `video_members: tuple[str, ...]`, kích thước nén/giải nén bằng bytes; root phải là `video/`. |
| `ZipArchiveExtractor` | `ArchiveInspection` + `lot_id` | `Path` source root | `video/` đổi thành `source/<lot_id>/`; không đổi tên video gốc. |
| `VideoDiscovery` | Source root + extension tuple | `VideoAsset[]` | `video_id` lấy từ filename stem, ví dụ `L21_V030.mp4` → `L21_V030`. |
| `ShotBoundaryPipeline` | `VideoAsset[]` + `ShotBoundaryConfig` | `scene-segments/<video_id>.json` + report | TransNetV2 chạy theo từng video; output gồm `segments`, `start_time/end_time`, threshold và backend; manifest hợp lệ được cache để resume. |
| `MetadataProvider` | `metadata/<video_id>.json` | JSON `Mapping` | `fps` là `float` tùy chọn; metadata nằm ngoài ZIP. |
| FFmpeg probe | File video | `VideoInfo` | `duration_ms: int`, `fps: float \| None`, `width/height: int`, `frame_count: int \| None`, `codec: str \| None`. Không có resolution/FPS cố định. |
| Frame scanner | Video + `VideoInfo` | `FrameCandidate[]` + decoder timeline cache | Mỗi candidate có `source_frame_number: int`, `timestamp_ms: int`, `pts_time_seconds: float`; `showinfo` được stream trực tiếp và cache theo fingerprint, không capture/replay stderr hoặc suy ra từ `nb_frames`/FPS. |
| Selection strategy | `VideoInfo` + candidates | `SelectedFrame[]` | `interval_ms`, `start_ms`, `end_ms` dùng milliseconds; số frame là `int`. |
| FFmpeg renderer | Video + selected frames | JPG/PNG + rendered manifest | Mặc định `target_short_edge_px=null`: giữ nguyên kích thước decoded; nếu đặt giá trị thì resize cạnh ngắn; frame ID là chuỗi sáu chữ số. |
| Video validator | VideoAsset + metadata + ProcessingResult | `ValidationReport` | Kiểm tra file, media, decode checkpoint, ảnh, dimensions, mapping và fingerprint; checkpoint là tỷ lệ `0.0..1.0`. |
| `PECoreEmbeddingPipeline` | `keyframes/<video_id>/*.{jpg,jpeg,png}` | `.npy` từng frame + `provenance.json`/video | Ảnh RGB được transform vào tensor `(B, 3, 448, 448)` theo PECore; `embedding.batch_size` là số ảnh/inference batch; `embedding.dataloader` điều chỉnh worker/pin/prefetch; encoder trả `(B, 1280)`, `float32`; mỗi file là `(1280,)`, L2 norm gần `1.0`. Pipeline vẫn đọc sidecar `.npy.meta.json` legacy nhưng output mới gộp provenance theo video để giảm small-file/fsync. |
| `KaggleStagingStrategy` | Keyframes, features, metadata, manifests, scene segments | `data/<lot_id>/kaggle-staging/` | Mặc định mỗi lot một snapshot độc lập; chỉ là file allowlist, không chứa archive, source hoặc video. Scene segment được thêm khi `upload.include_scene_segments=true`. |
| `CumulativeKaggleStagingStrategy` | Keyframes, features, metadata, manifests, scene segments của lot mới | `data/kaggle-dataset-staging/` | Legacy opt-in: merge idempotent theo `video_id`, giữ dữ liệu các lot trước và không chứa archive/source/video. |
| `KaggleCliUploader` | `StagingResult` | `UploadResult` | `mode=auto` tự chọn `create`/`version` theo dataset ref; `verified: bool`; `dir_mode` là `skip`, `zip` hoặc `tar`. |
| `CleanupManager` | UploadResult đã verify | `CleanupResult` | Xóa các directory được bật trong config; metadata độc lập, reports, receipts và state vẫn giữ lại. |

### Traceback tiến độ và hiệu năng

Checklist triển khai/kiểm tra của đợt refactor nằm tại `preprocess/plan.md`.
Runtime không ghi vào checklist này; runtime trace được chia theo mục đích:

```text
data/<lot_id>/state.json                 # stage + checkpoint từng video
data/<lot_id>/upload-state.json          # staging/upload/cleanup độc lập
data/<lot_id>/reports/processing.json    # artifact từng video đã validate
data/<lot_id>/reports/embedding.json     # feature từng video + device provenance
data/<lot_id>/reports/decoder-timelines/ # cache FFmpeg frame n + PTS
data/<lot_id>/reports/upload-disk-budget.json
data/<lot_id>/receipts/upload.json       # remote status/inventory evidence
```

Mỗi stage ghi `attempt`, `started_at`, `completed_at`, `elapsed_seconds` và
fingerprint. Fingerprint runtime được scope theo stage: thay docs không làm
TransNet/render chạy lại; source code chưa commit vẫn được nhận qua content
digest của module liên quan. `scene-segments` là output của TransNet nhưng là
input của `process_validate`, nên tạo boundary lần đầu không tự invalidate
stage TransNet ở lần chạy kế tiếp.

Nếu FFmpeg render nhanh trả về ít PNG hơn số frame được chọn (thường do VFR,
timestamp không hợp lệ hoặc decoder ordinal khác ffprobe), extractor không bỏ
qua phần thiếu. Nó đọc lại timeline `frame n + PTS` bằng chính FFmpeg decoder,
ánh xạ lại theo PTS và render lần hai; `FrameRef`/frame ID trong selection
manifest vẫn được giữ nguyên. Nếu PTS không thể ánh xạ hoặc lần hai vẫn thiếu
frame, pipeline dừng để validation không nhận một dataset không đầy đủ.

Các thông số chính được cấu hình trong `config.json`:

```text
download.max_concurrent_connections = 16   # connection/server
download.split_count                  = 16   # segment/archive
processing.interval_ms                = 1000 # milliseconds
processing.target_short_edge_px       = null # preserve decoded dimensions
processing.png_compress_level         = 6    # FFmpeg/Pillow PNG level 0..9
processing.ffmpeg_threads             = 0    # 0=FFmpeg auto; không gắn số core/GPU
processing.decode_checkpoints          = [0, 0.5, 1] # normalized position
embedding.expected_dim                = 1280 # vector values
embedding.batch_size                  = 8    # images/encoder call
embedding.precision                   = fp32 # model load: fp32/fp16/bf16
embedding.autocast.enabled            = false # AMP around encode_image
embedding.autocast.dtype              = bf16 # AMP dtype: fp16/bf16
embedding.autocast.cache_enabled      = true
embedding.tf32.enabled                = false # CUDA TF32 for eligible FP32 ops
embedding.tf32.matmul                 = true  # CUDA matmul TF32 flag
embedding.tf32.cudnn                  = true  # cuDNN convolution TF32 flag
embedding.dataloader.num_workers     = 0    # CPU image preprocessing workers
embedding.dataloader.pin_memory      = false # host→CUDA transfer hint
embedding.dataloader.persistent_workers = false
embedding.dataloader.prefetch_factor = 2    # only when num_workers > 0
embedding.model_revision              = null # pin HF commit in strict mode
upload.mode                           = auto # create lần đầu, version khi dataset đã có
upload.staging_scope                  = lot  # một staging độc lập cho mỗi lot
upload.dataset_ref_template           = owner/aic2026-hcmc-{lot_slug}
upload.metadata_template              = data/kaggle-dataset-metadata.json
upload.require_remote_inventory       = true # status + tải/đối chiếu provenance remote
upload.missing_artifact_policy        = error # error hoặc skip artifact tùy chọn
upload.temporary_space_multiplier     = 1.0 # reserve cho package tạm của CLI
reproducibility.mode                  = best_effort # strict để pin seed/CUBLAS
reproducibility.seed                  = 2026
progress.enabled                      = true # terminal/tmux progress bars
progress.leave                        = false # clear completed bars; less terminal noise
progress.bar_width                    = 24   # fixed width of each bar
shot_boundary.enabled                 = true # auto-run when selector needs scenes
shot_boundary.backend                 = transnetv2
shot_boundary.device                  = auto # auto/cpu/cuda/mps
shot_boundary.threshold               = 0.5
shot_boundary.window_batch_size       = 1 # TransNet windows/GPU call; tự tune theo host
shot_boundary.overwrite               = false # reuse valid manifests
scheduling.overlap_upload             = false # lot N upload cùng lúc xử lý lot N+1
scheduling.overlap_render_embedding   = false # render video N+1 khi embed video N
scheduling.max_pending_uploads        = 1 # hiện cố định 1 để bound disk
scheduling.max_pending_embeddings     = 1 # 1..10 video running/queued; một GPU worker
```

Progress bar dùng `tqdm` và hiển thị tối đa bảy dòng cố định trong terminal/tmux:
dòng lot/stage, dòng CPU/GPU, dòng disk, full-pipeline bar, detail bar của
operation hiện tại, embedding bar và upload bar. Hai dòng cuối chỉ xuất hiện
khi activity tương ứng bắt đầu và giữ vị trí riêng, vì vậy upload nền, embedding
GPU và render foreground không vẽ đè nhau. Các operation lồng nhau dùng lại
detail bar nên không tạo thêm hàng. Các progress bar hiển thị theo dạng
`elapsed<eta (rate)`, ví dụ `00:12<01:45 (8.4 video/s)`; `bar_width` là chiều
dài tối đa của phần bar và tự co lại nếu terminal hẹp; `leave=false` xóa các
bar khi pipeline kết thúc. Embedding bar đếm video đã hoàn tất/khôi phục;
upload bar đọc byte progress của Kaggle CLI và đổi sang trạng thái `verifying`
rồi `verified` sau khi transfer kết thúc.
Khi output đi qua `tee`, pipeline tự chuyển sang một dòng ASCII duy nhất để
không ghi cursor escape code vào log, nhưng vẫn giữ full-pipeline, operation
hiện tại, embedding/upload, CPU/GPU/disk metrics và ETA. Để xem layout nhiều
dòng cố định, chạy
trực tiếp trong terminal/tmux không pipe output qua `tee`. Warning CUDA không
gây lỗi pipeline. Với `reproducibility.mode=best_effort`, pipeline lọc warning
CUDA lặp lại để terminal không bị rác; chế độ này không cam kết kết quả
bit-level. Với `reproducibility.mode=strict`, pipeline đặt
`CUBLAS_WORKSPACE_CONFIG`, seed và deterministic mode trước khi load model.
Muốn pin model Hugging Face trong strict mode, phải đặt
`embedding.model_revision` thành commit SHA 40 ký tự; branch/tag như `main`
không được coi là immutable.

`embedding.precision` chỉ kiểm tra tên dtype khi đọc config; capability được
kiểm tra sau khi `device=auto` đã resolve. `bf16` trên CUDA sẽ dừng với lỗi rõ
ràng nếu GPU không hỗ trợ bfloat16. `fp16` hoặc `bf16` trên CPU được chuyển
xuống backend để backend quyết định; MPS hiện yêu cầu `precision=fp32`.

`autocast` là AMP inference độc lập với precision của model:

```text
precision=fp32 + autocast.enabled=true  → model FP32, phép tính AMP theo dtype
precision=fp16/bf16 + autocast=false     → model được load ở low precision
```

Không có profile mặc định cho RTX 3060 hoặc dung lượng VRAM cụ thể. Một profile
CUDA thường dùng là `precision=fp32` kết hợp `autocast.enabled=true`; chỉ đặt
`autocast.dtype=bf16` nếu chính GPU được chọn hỗ trợ, nếu không dùng `fp16`.
`torch.inference_mode()` chỉ tắt gradient, không phải AMP. `pin_memory=true`
chỉ được áp dụng thực tế khi device đã resolve thành CUDA; trên CPU/MPS nó tự
tắt để không tạo overhead vô ích.

`tf32` là backend flag CUDA độc lập với `precision` và `autocast`. Nó không
đổi dtype của model hoặc vector output, mà cho phép các phép toán FP32 đủ điều
kiện dùng TensorFloat-32:

```text
embedding.tf32.enabled=true
  → torch.backends.cuda.matmul.allow_tf32 = true
  → torch.backends.cudnn.allow_tf32 = true
```

Chỉ áp dụng cấu hình này sau khi `device=auto` đã resolve thành CUDA; trên
CPU/MPS nó được bỏ qua để giữ config portable. CUDA dưới compute capability 8.0 sẽ dừng
với lỗi rõ ràng khi TF32 được bật. Với `reproducibility.mode=strict`, nên để
`embedding.tf32.enabled=false` vì TF32 có thể làm khác kết quả giữa các GPU.
Thay đổi TF32 cũng làm encoder cache fingerprint thay đổi.

Nếu cần chọn GPU vật lý mà không sửa code, dùng CUDA visibility trước lệnh:

```bash
CUDA_VISIBLE_DEVICES=1 \
python -m preprocess.batch --config preprocess/batch/config.json run
```

Trong process, GPU đó sẽ được nhìn thấy là `cuda`. Pipeline cũng nhận
`cuda:1` trực tiếp trong `embedding.device` và `shot_boundary.device`; preflight
sẽ fail trước download nếu index không tồn tại. `CUDA_VISIBLE_DEVICES` vẫn hữu
ích khi muốn cô lập process khỏi các GPU khác.

`shot_boundary.window_batch_size` batch các cửa sổ TransNetV2 100 frame, bước
50 frame. Giá trị `1` giữ đường inference gốc và dùng ít VRAM nhất. Tăng dần
`2`, `4`, `8`, ... chỉ sau khi kiểm tra VRAM và so sánh scene output; pipeline
không tự chọn theo tên GPU. Adapter probe FPS một lần/video và giải phóng tensor
tạm sau video. AMP/low precision cho TransNetV2 không bật vì có thể đổi boundary
sát threshold; PE-Core có config precision/autocast riêng và đã kiểm tra
capability.

Rule `linear-rulebase` dùng milliseconds và số frame:

```text
shot <= 1s       → 1 frame
shot <= 3s       → 2 frames
mỗi thêm 3s      → thêm 1 frame
```

### Cài tool trên Ubuntu

`aria2c`, `ffmpeg` và `ffprobe` là system executable. Các Python dependency
(gồm Kaggle CLI, TransNetV2 PyTorch và open_clip_torch cho PE-Core) được cài riêng trong
`preprocess/.venv`, không dùng chung environment của phần khác. Trên Ubuntu,
chạy từ repository root:

```bash
sudo apt-get update
sudo apt-get install -y aria2 ffmpeg tmux python3 python3-venv python3-pip ca-certificates

python3 -m venv preprocess/.venv
source preprocess/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r preprocess/requirements.lock.txt
```

Kiểm tra các executable trước khi chạy pipeline:

```bash
command -v aria2c
command -v ffmpeg
command -v ffprobe
command -v kaggle

aria2c --version
ffmpeg -version
ffprobe -version
kaggle --help
```

Nếu đã có virtual environment riêng cho `preprocess`, chỉ cần activate nó rồi
chạy lại `python -m pip install -r preprocess/requirements.lock.txt`.
`requirements.txt` là bản cài đặt tương thích tương ứng; dùng
`requirements.lock.txt` khi muốn giữ đúng các phiên bản top-level đã kiểm thử.
Đây là top-level lock, chưa phải lock có hash của toàn bộ dependency transitives;
`provenance.json` ghi phiên bản thực tế. Muốn tái lập nghiêm ngặt phải giữ thêm
image/container, CUDA driver và binary FFmpeg/aria2 của host.
`aria2c`
không nằm trong Python requirements vì được cài từ APT. Cách quản lý package
bằng APT theo tài liệu Ubuntu và cách cài/auth Kaggle CLI xem
[Ubuntu package management](https://documentation.ubuntu.com/server/how-to/software/package-management/)
và [Kaggle CLI documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md).

Kaggle cần authentication trước khi bật upload. Cách tương thích với cả
Kaggle CLI 1.x (Python 3.10) và 2.x là tải `kaggle.json` từ trang API của
Kaggle rồi đặt vào `~/.kaggle/kaggle.json`; không ghi file này vào repository:

```bash
mkdir -p ~/.kaggle
# chép file kaggle.json đã tải từ trang Kaggle API vào ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets list
```

Nếu đang dùng Python 3.11+ và Kaggle CLI 2.x, có thể dùng OAuth:

```bash
kaggle auth login
```

Trên host Python 3.10, không dùng `kaggle auth login` vì CLI 1.x không có
subcommand này; dùng `kaggle.json` như trên.

### Quick run từ SSH (không thay thế runbook tmux)

Tạo config từ file mẫu và chỉnh tool/path theo máy SSH. Path tương đối trong
config được giải thích tương đối với thư mục chứa file config:

```bash
cp preprocess/batch/config.example.json preprocess/batch/config.json
mkdir -p data/input data/metadata

python -m preprocess.batch --config preprocess/batch/config.json preflight
python -m preprocess.batch --config preprocess/batch/config.json parse-links
python -m preprocess.batch --config preprocess/batch/config.json run
```

Với config mẫu đặt tại `preprocess/batch/config.json`, các path
`../../data/...` trỏ về `<repository-root>/data/...`. Kiểm tra cách resolve
trước khi chạy:

```bash
python -c 'from pathlib import Path; from preprocess.batch.config import BatchConfig; c=BatchConfig.from_json(Path("preprocess/batch/config.json")); print("data_root=", c.data_root); print("links_file=", c.links_file); print("metadata_root=", c.metadata_root); print("scene_segments_dir=", c.processing.scene_segments_dir); print("metadata_template=", c.upload.metadata_template)'
test -f data/input/links.txt
```

Nếu config dùng path tuyệt đối như `/data`, mọi input/output sẽ nằm dưới
`/data`, không nằm dưới `./data`; hãy kiểm tra đúng mount point trước khi
chạy.

Nếu virtual environment không nằm trong `PATH`, thay `python` bằng Python của
environment đó. `preflight` kiểm tra `aria2c`, `ffmpeg`/filter `showinfo`, `ffprobe`, thêm
TransNetV2 khi selector cần scene, thêm `kaggle` khi upload được bật, và kiểm
tra dung lượng trống trước khi tải.

Khi bật `upload.enabled`, tạo file metadata Kaggle tại path cấu hình bởi
`upload.metadata_template` (mặc định trong config mẫu là
`data/kaggle-dataset-metadata.json`). Bạn không cần tự tạo từng dataset trên
trang Kaggle. Với cấu hình mặc định, `upload.dataset_ref_template` được bind
theo lot: `L22_a` trở thành ví dụ `owner/aic2026-hcmc-l22-a`. `mode=auto` gọi
`kaggle datasets status`; nếu dataset chưa tồn tại pipeline chạy `datasets
create`, nếu đã tồn tại pipeline chạy `datasets version`.

Chỉ cấu hình **một** trong `upload.dataset_ref` và
`upload.dataset_ref_template`. `owner` trong template là username Kaggle của
bạn. File metadata template chỉ
là mẫu; trước khi upload pipeline tự ghi `id` của lot vào
`kaggle-staging/dataset-metadata.json`. Nếu dùng `upload.dataset_ref` cố định
thay cho template, mọi lot sẽ dùng chung một dataset remote.

Staging được dựng trong thư mục tạm rồi đổi tên atomically. Mỗi staging có
`provenance.json` chứa config, revision code, phiên bản package/tool, model
cache fingerprint và SHA-256 inventory của payload. Khi
`require_remote_inventory=true`, upload chỉ được coi là verify sau khi status
là `ready`, pipeline tải chính xác file remote `provenance.json` bằng Kaggle CLI
và đối chiếu `payload_digest` với staging local. Cách này không phụ thuộc giới
hạn 200 kết quả của `kaggle datasets files`, kể cả dataset có hàng nghìn `.npy`.
Trong cùng lệnh upload, payload digest được tái sử dụng và cleanup tin receipt
vừa verify để tránh đọc lại hàng chục GB. Lệnh `cleanup` chạy độc lập vẫn audit
inventory local và tải lại provenance remote trước khi xóa.

Nếu SSH bị ngắt hoặc verifier cũ báo lỗi sau khi transfer đã hoàn tất, chạy lại
`run` hoặc `upload --lot-id ...`: trước khi gửi dữ liệu, uploader kiểm tra status
và provenance remote. Nếu digest đã trùng, transfer được bỏ qua, receipt được
khôi phục và pipeline tiếp tục cleanup; nếu digest khác thì mới tạo version mới.

Không bật upload khi chưa kiểm tra credentials bằng `kaggle datasets list`.

Nếu các stage xử lý đã hoàn tất nhưng chỉ muốn retry/cập nhật upload của một
lot, dùng lệnh riêng:

```bash
python -m preprocess.batch \
  --config preprocess/batch/config.json \
  upload --lot-id L29_a
```

Lệnh này không chạy lại download, extraction, TransNetV2, keyframe hoặc
embedding. Nó đọc artifact đã validate, ghi trạng thái upload riêng vào
`data/L29_a/upload-state.json`, và dùng `data/L29_a/upload.lock` để không cho
hai tiến trình upload cùng lot chạy đồng thời.

`upload --lot-id` là lệnh explicit nên không yêu cầu `upload.enabled=true`; cờ
này chỉ quyết định `run` có tự động upload hay không. `dataset_ref` hoặc
`dataset_ref_template`, `metadata_template` và các tham số Kaggle khác vẫn
phải hợp lệ. Nếu
`upload-state.json` đã ghi cả `stage_upload` và `cleanup` là `completed`, lệnh
sẽ dùng receipt đã lưu, không đọc lại artifact local và không bị ảnh hưởng bởi
việc đổi config sau đó.

Nếu upload đã verify nhưng cleanup bị ngắt hoặc bị tắt lúc đầu, chạy cleanup
riêng; lệnh sẽ re-verify dataset remote trước khi xóa:

```bash
python -m preprocess.batch \
  --config preprocess/batch/config.json \
  cleanup --lot-id L29_a
```

Cleanup không bao giờ xóa chỉ vì có file `upload.json`; receipt phải là upload
đã verify và remote phải còn ở trạng thái `ready`. Các stage upload/cleanup và
receipt của chúng nằm trong `upload-state.json`, tách khỏi checkpoint xử lý.

Ở cấu hình mặc định, lệnh trên tạo hoặc cập nhật **dataset riêng của lot**.
Sau khi Kaggle trả trạng thái verify thành công, `kaggle-staging/` của lot được
cleanup theo `cleanup.delete_staging`; dữ liệu remote của các lot trước không
cần có trên máy SSH.

Nếu thực sự muốn dùng một dataset cumulative, phải bật rõ chế độ legacy bằng
`upload.staging_scope=dataset` và dùng `upload.dataset_ref` cố định. Khi đó
pipeline mới giữ `data/kaggle-dataset-staging/` rồi upload toàn bộ snapshot
tích lũy; chế độ này không phải mặc định.

Không chạy đồng thời hai lệnh upload ở các lot khác nhau khi dùng legacy
cumulative mode vì chúng dùng chung dataset-level lock.

Có thể kiểm tra một ZIP đã tải mà không chạy cả pipeline:

```bash
python -m preprocess.batch --config preprocess/batch/config.json \
  inspect-archive data/L29_a/archive/Videos_L29_a.zip
```

### Runbook SSH hoàn chỉnh với tmux

Workflow dưới đây giữ pipeline chạy sau khi kết nối SSH bị ngắt. Chỉ kết thúc
tmux sau khi lệnh `run` đã trả prompt và `state.json` của các lot là
`completed`.

#### 1. Kết nối và vào repository

Từ máy local:

```bash
ssh -i ~/.ssh/id_ed25519 <user>@<ssh-host>
```

Trên SSH server:

```bash
cd /path/to/AIC-HCMC-2026
git switch preprocess
git pull --ff-only origin preprocess
git branch --show-current       # kết quả phải là preprocess
```

Không push hoặc merge trong workflow này. Mọi thay đổi code vẫn chỉ thuộc
branch `preprocess`. Pull code trước khi thêm option mới vào config; nếu config
có `scheduling.overlap_render_embedding` nhưng source vẫn ở `main`/bản cũ,
Python sẽ báo `unexpected keyword argument` trước khi pipeline chạy.

#### 2. Cài system tools và Python environment

Chạy một lần trên máy mới:

```bash
sudo apt-get update
sudo apt-get install -y aria2 ffmpeg tmux python3 python3-venv python3-pip ca-certificates

python3 -m venv preprocess/.venv
preprocess/.venv/bin/python -m pip install --upgrade pip
preprocess/.venv/bin/python -m pip install -r preprocess/requirements.lock.txt
```

Kiểm tra:

```bash
command -v aria2c ffmpeg ffprobe tmux
preprocess/.venv/bin/python -m pip check
```

`aria2c`, `ffmpeg`, `ffprobe` và `tmux` là system tools. Python
packages được cài riêng trong `preprocess/.venv`. Lần đầu PE-Core chạy có thể
tải model weights nhiều GB vào cache Hugging Face.

#### 3. Chuẩn bị input và config

```bash
mkdir -p data/input data/metadata data/logs
test -f preprocess/batch/config.json || \
  cp preprocess/batch/config.example.json preprocess/batch/config.json
nano data/input/links.txt
nano preprocess/batch/config.json
```

`data/input/links.txt` có một URL ZIP mỗi dòng:

```text
https://aic-data.ledo.io.vn/Videos_L29_a.zip
https://aic-data.ledo.io.vn/Videos_L30_b.zip
```

Đặt metadata độc lập theo video:

```text
data/metadata/L21_V030.json
data/metadata/L21_V031.json
```

Trong config, kiểm tra tối thiểu:

```json
{
  "download": {
    "max_concurrent_connections": 16,
    "split_count": 16
  },
  "shot_boundary": {
    "enabled": true,
    "backend": "transnetv2",
    "device": "auto",
    "threshold": 0.5,
    "window_batch_size": 1,
    "overwrite": false
  },
  "processing": {
    "selector": "linear-rulebase",
    "scene_segments_dir": "../../data/scene-segments",
    "ffmpeg_threads": 0
  },
  "embedding": {
    "enabled": true
  },
  "upload": {
    "enabled": true,
    "mode": "auto",
    "dir_mode": "zip",
    "staging_scope": "lot",
    "dataset_ref_template": "owner/aic2026-hcmc-{lot_slug}",
    "metadata_template": "../../data/kaggle-dataset-metadata.json",
    "include_scene_segments": true,
    "require_remote_inventory": true,
    "missing_artifact_policy": "error"
  },
  "scheduling": {
    "overlap_upload": false,
    "overlap_render_embedding": true,
    "max_pending_uploads": 1,
    "max_pending_embeddings": 10
  }
}
```

Nếu chỉ muốn tạo artifact local, đặt `upload.enabled=false`; khi đó pipeline
không tự cleanup. Nếu bật upload, file JSON nguồn tại
`upload.metadata_template` phải tồn tại đúng đường dẫn và
`dataset_ref_template` phải có owner Kaggle hợp lệ. Tên file nguồn trong config
mẫu là `kaggle-dataset-metadata.json`; pipeline luôn tạo bản dành cho Kaggle
trong staging với tên cố định `dataset-metadata.json`. Không cần tạo dataset thủ
công; `mode=auto` tự chọn create/version.

#### 4. Xác thực Kaggle

Chỉ cần thực hiện khi `upload.enabled=true`:

```bash
export KAGGLE_API_TOKEN='<your-kaggle-api-token>'
kaggle datasets list
```

Không ghi token vào `config.json` hoặc commit vào repository.

#### 5. Tạo tmux session

```bash
tmux new -s aic-preprocess
```

Tất cả lệnh xử lý tiếp theo chạy bên trong session này:

```bash
cd /path/to/AIC-HCMC-2026
source preprocess/.venv/bin/activate
set -o pipefail
```

#### 6. Preflight và kiểm tra input

Chạy preflight trước khi tải:

```bash
python -m preprocess.batch \
  --config preprocess/batch/config.json \
  preflight 2>&1 | tee data/logs/preflight.log
```

Kiểm tra các archive request mà parser tạo ra:

```bash
python -m preprocess.batch \
  --config preprocess/batch/config.json \
  parse-links 2>&1 | tee data/logs/parsed-links.json
```

Nếu preflight hoặc parse-links lỗi, sửa lỗi trước khi chạy batch.

#### 7. Chạy toàn bộ pipeline

```bash
python -m preprocess.batch \
  --config preprocess/batch/config.json \
  run
```

Chạy trực tiếp như trên để thấy các row pipeline/detail/embedding/upload cố
định trong tmux. Nếu cần đồng thời ghi stdout/stderr bằng `tee`, dùng lệnh dưới;
reporter sẽ tự chuyển sang một dòng ASCII để file log không chứa cursor escape:

```bash
python -m preprocess.batch \
  --config preprocess/batch/config.json \
  run 2>&1 | tee data/logs/batch-run.log
```

Nếu muốn các lot sau vẫn chạy khi một lot lỗi, thêm
`run --continue-on-error`. Lệnh sẽ trả exit code `1` nếu còn lot lỗi và in
`failed_lots` ở JSON cuối; checkpoint của lot lỗi vẫn giữ nguyên để retry.

Một lot sẽ đi theo thứ tự:

```text
download → archive validate → unzip → discover video
→ TransNetV2 shot boundaries → select/render keyframe → validate
→ PECore embedding nếu bật
→ Kaggle staging → upload/status verify
→ cleanup nếu upload đã verify
```

Không đóng terminal SSH hoặc kill tmux khi lệnh còn chạy.

#### 8. Detach, reconnect và theo dõi

Detach khỏi tmux nhưng giữ process chạy:

```text
Ctrl-b rồi nhấn d
```

Sau đó có thể đóng SSH:

```bash
exit
```

Kết nối lại:

```bash
ssh -i ~/.ssh/id_ed25519 <user>@<ssh-host>
cd /path/to/AIC-HCMC-2026
tmux ls
tmux attach -t aic-preprocess
```

Hoặc theo dõi log từ một SSH connection khác:

```bash
cd /path/to/AIC-HCMC-2026
tail -f data/logs/batch-run.log
```

Xem 50 dòng cuối của tmux pane:

```bash
tmux capture-pane -p -t aic-preprocess -S -50
```

Kiểm tra checkpoint của một lot:

```bash
grep -n '"state"' data/L29_a/state.json
tail -n 40 data/L29_a/state.json
tail -n 40 data/L29_a/upload-state.json
```

Khi selector cần scene, checkpoint sẽ đi qua `shot_boundaries` rồi
`shot_boundaries_ready` trước `processing`. File boundary được lưu ngoài lot
ở `data/scene-segments/<video_id>.json`; nếu stage bị ngắt, file hợp lệ sẽ
được dùng lại ở lần chạy sau khi `shot_boundary.overwrite=false`.

Mỗi lot còn có `state.json.stages` với các stage `running` hoặc `completed`.
Stage `process_validate` có thêm
`state.json.stages.process_validate.videos.<video_id>`; mỗi video được đánh dấu
`running` trước khi xử lý và `completed` sau khi render + validate thành công,
kèm fingerprint, artifact result và thời lượng. Stage `embedding` cũng có
`state.json.stages.embedding.videos.<video_id>`; do đó ngắt sau video nào thì
vector video đó được validate/restore thay vì infer lại. PE-Core provenance mới
được gom tại `dataset/PECore-features/<video_id>/provenance.json`; sidecar legacy
vẫn đọc được. Khi `scheduling.overlap_render_embedding=true`, worker GPU còn
ghi journal atomically tại
`reports/embedding-completions/<video_id>.json`. Worker không ghi `state.json`;
main thread nhận kết quả rồi mới cập nhật checkpoint và report tổng. Nếu SSH bị
ngắt đúng khoảng giữa hai thao tác này, lần chạy sau validate journal và nhận
lại feature thay vì infer lại. Checkpoint được ghi trước và sau từng stage cũng như từng video;
nếu SSH bị mất hoặc nhấn `Ctrl-C`, lần chạy
`run` tiếp theo sẽ:

```text
stage đã completed + artifact còn hợp lệ  → restore/skip
video đã completed + artifact còn hợp lệ   → restore/skip video đó
stage/video đang running hoặc artifact hỏng → chạy lại phần tương ứng
```

Nếu một lot đã có `state.json` với trạng thái `completed`, lần chạy `run` sau
sẽ tự động bỏ qua lot đó và tiếp tục lot kế tiếp trong `links.txt`. Request mới
phải vẫn trùng URL/archive đã ghi trong checkpoint; nếu dùng cùng `lot_id` cho
một archive khác, pipeline sẽ dừng để tránh ghi đè dữ liệu.

Các artifact được kiểm tra lại trước khi skip: ZIP, source video, scene
manifest, discovery report, rendered/validation manifest, `.npy` feature và
upload/cleanup receipt. Fingerprint theo stage, video và dependency chain cũng
cho phép đổi riêng `embedding.dataloader` mà vẫn giữ lại download, extraction,
TransNetV2 và processing hợp lệ; embedding và các stage phụ thuộc nó sẽ chạy
lại. Khi stage hoặc video hoàn tất, `state.json` ghi `started_at`,
`completed_at` và `elapsed_seconds`; event history cũng lưu thời lượng tương
ứng. Không xóa `state.json` để resume.

`CheckpointStore` ghi `state.json` atomically nên việc đọc file trong lúc pipeline
đang chạy sẽ thấy bản cũ hoặc bản mới hoàn chỉnh, không thấy JSON dở dang. Không
chạy hai pipeline xử lý cùng lot, và không chỉnh sửa thủ công các file state khi
process đang hoạt động. Các thao tác upload có lock riêng theo lot tại
`upload.lock`; vì vậy không chạy hai lệnh upload cùng lot song song, cũng không
chạy `upload --lot-id <lot>` song song với `run` cho cùng lot. Mặc định pipeline
chính upload tuần tự sau processing/embedding của từng lot. Nếu
`scheduling.overlap_render_embedding=true`, sau khi video N render + validate,
pipeline submit task embed N vào một GPU worker rồi tiếp tục render N+1 bằng
FFmpeg/CPU. `max_pending_embeddings` nhận giá trị `1..10` và đếm tổng task đang
chạy hoặc đang nằm trong FIFO queue. Khi queue đạt giới hạn, foreground nhận
xong task cũ nhất trước khi submit thêm, nên không mất video và backlog không
tăng vô hạn. Giá trị `10` cho phép renderer đi trước tối đa mười task GPU; nó
không tạo mười model/worker chạy đồng thời. Keyframe đã embed vẫn được giữ để
upload nên tổng dung lượng keyframe của lot không giảm bởi option này. Progress
nội bộ của worker bị tắt để không tranh terminal; orchestrator cập nhật một
embedding bar riêng theo số video worker hoàn tất/restore. Detail bar foreground
dùng nhãn `render+embed`, còn GPU utilization/memory vẫn được system metrics cập
nhật. Custom embedding strategy có mutable progress/UI
nên override `BatchEmbeddingStrategy.for_background()` để trả worker view độc
lập. PECore worker overlap buộc DataLoader `num_workers=0` dù config đặt lớn
hơn: tạo process bằng `fork` từ background thread trên Linux có thể deadlock.
`batch_size`, `pin_memory`, precision/autocast và model vẫn giữ nguyên; nếu cần
DataLoader nhiều process thì tắt overlap để dùng embedding stage tuần tự.

Nếu
`scheduling.overlap_upload=true`, upload/cleanup lot N chạy trong một worker
nền trong khi lot N+1 được xử lý; chỉ một upload pending được phép, và dataset
cumulative vẫn serialize bằng dataset lock. Chế độ này giữ tối đa artifact của
lot đang xử lý cộng một lot đang upload, nên phải để đủ disk reserve. Progress
nền cập nhật upload bar riêng, không vẽ đè detail/embedding bar;
`state.json`/`upload-state.json` vẫn tách biệt.

Hai overlap có thể bật cùng lúc: trong lot hiện tại CPU render song song GPU
embedding, đồng thời network upload lot trước. Nếu disk, CPU preprocessing hoặc
network cùng tranh tài nguyên, hãy tắt `overlap_upload` trước; render–embed chỉ
giữ tối đa `max_pending_embeddings` task chạy/chờ và thường là phần có lợi trực
tiếp hơn.

`run.lock` bảo vệ pipeline chính; upload/cleanup dùng thêm `upload.lock`. Với
legacy cumulative staging, `kaggle-dataset-upload.lock` bảo vệ dataset chung và
`kaggle-dataset-transaction.json` cho phép khôi phục directory swap nếu SSH bị
ngắt giữa lúc merge. Resume dùng size + mtime làm fast path trên cùng filesystem;
khi timestamp đổi hoặc full audit được yêu cầu, SHA-256 là nguồn xác nhận nội
dung. SHA file lớn được cache theo inode/size/mtime/ctime trong một process.

Trạng thái hai stage `stage_upload` và `cleanup` nằm trong
`data/<lot_id>/upload-state.json`; `data/<lot_id>/state.json` giữ pipeline
chính và được chuyển sang `completed` sau khi upload độc lập thành công.

#### 9. Kiểm tra sau khi hoàn thành

Khi `run` đã kết thúc:

```bash
find data/L29_a -type f | sort
grep -n '"state"' data/L29_a/state.json
```

Trạng thái cuối phải là:

```text
"state": "completed"
```

Nếu upload được verify, receipt nằm tại:

```text
data/L29_a/receipts/upload.json
```

Nếu upload được verify và cleanup bật, `archive/`, `source/`, keyframes,
features, manifests và staging riêng của lot có thể đã bị xóa theo config.
Dataset remote của lot không cần giữ lại artifact local của các lot trước. Chỉ
legacy cumulative mode mới giữ `data/kaggle-dataset-staging/` làm base cho
version tiếp theo. Nếu upload tắt, các artifact local vẫn được giữ lại.

#### 10. Kết thúc tmux và SSH session

Chỉ chạy sau khi batch đã hoàn thành:

```bash
tmux kill-session -t aic-preprocess
exit
```

`tmux kill-session` sẽ dừng mọi process còn chạy trong session, nên không dùng
lệnh này để detach giữa chừng. Khi cần giữ process, dùng `Ctrl-b d`.

### PE-Core image embedding

Script/module: `preprocess/pecore/`. Đây là image encoder đầy đủ của
`PE-Core-bigG-14-448` qua `open_clip_torch`; model ONNX text-only không tạo
được image embedding.

Input và output giữ cùng quy ước với `AIC2026_sample`:

```text
data/L29_a/dataset/keyframes/L21_V030/000022.png
  → data/L29_a/dataset/PECore-features/L21_V030/000022.npy
  + data/L29_a/dataset/PECore-features/L21_V030/provenance.json
```

Vector output là `float32`, shape `(1280,)`, đã L2-normalize. Model chỉ được
load ở lần encode đầu tiên; các `.npy` hợp lệ được bỏ qua khi chạy lại. Lần
chạy đầu có thể tải nhiều GB model weights vào cache của Hugging Face.

Embed một video:

```bash
preprocess/.venv/bin/python -m preprocess.pecore \
  --input-root data/L29_a/dataset/keyframes \
  --output-root data/L29_a/dataset/PECore-features \
  --video-id L21_V030 \
  --device cpu \
  --batch-size 4 \
  --num-workers 2 \
  --prefetch-factor 2
```

Embed tất cả thư mục video dưới `--input-root` thì bỏ các cờ `--video-id`.
Dùng `--overwrite` khi muốn tính lại các feature đã tồn tại. `--pin-memory`
và `--persistent-workers` tương ứng với các cờ trong `embedding.dataloader`;
`--persistent-workers` yêu cầu `--num-workers` lớn hơn 0.
Worker hiện tồn tại trong iterator của một video; checkpoint theo video được
ưu tiên hơn việc giữ worker xuyên ranh giới video.
`--tf32` bật cả CUDA matmul TF32 và cuDNN TF32 trong standalone command.

Để chạy tự động sau bước validate trong batch pipeline, bật:

```json
{
  "embedding": {
    "enabled": true,
    "model_id": "hf-hub:timm/PE-Core-bigG-14-448",
    "model_revision": "<40-character-hugging-face-commit-sha>",
    "device": "auto",
    "precision": "fp32",
    "autocast": {
      "enabled": true,
      "dtype": "bf16",
      "cache_enabled": true
    },
    "tf32": {
      "enabled": true,
      "matmul": true,
      "cudnn": true
    },
    "expected_dim": 1280,
    "batch_size": 8,
    "dataloader": {
      "num_workers": 0,
      "pin_memory": false,
      "persistent_workers": false,
      "prefetch_factor": 2
    },
    "overwrite": false,
    "features_dir_name": "PECore-features"
  }
}
```

`model_revision` là tùy chọn ở `best_effort`; ở `strict` với model
`hf-hub:` nó phải là commit SHA 40 ký tự. Mỗi thư mục video có một
`provenance.json` ánh xạ frame sang SHA-256 ảnh nguồn và fingerprint encoder;
sidecar `.npy.meta.json` cũ vẫn được đọc để migration. Vì vậy đổi
model/precision, autocast, TF32 hoặc sửa ảnh sẽ không bị nhầm là cache hợp lệ. Với
`embedding.dataloader.num_workers > 0`,
`pin_memory=true` chỉ được dùng khi device resolve thành CUDA, không thay đổi
số lượng vector hay thứ tự frame.

TransNetV2 hiện giữ FP32 và `torch.no_grad()`. Package
`transnetv2-pytorch` không expose precision/autocast trong API mà pipeline đang
dùng; hơn nữa scene boundary nhạy với sai khác số học. Vì vậy nên tối ưu
TransNetV2 bằng `device=cuda`, cache manifest và các memory setting của chính
package trước. Chỉ nên bổ sung `inference_dtype`/autocast riêng cho TransNetV2
sau khi có regression so sánh boundary trên một tập video cố định; không dùng
chung `embedding.precision` cho detector.

Khi `embedding.enabled=true`, pipeline thực hiện:

```text
validated keyframes → PECore-features/*.npy → Kaggle staging → upload verify
```

Encoder và pipeline đều là dependency-injected; có thể thay
`VisualEmbeddingEncoder` bằng model/strategy khác mà không đổi writer, layout
hay Kaggle staging.

### Artifact và validate

Mặc định selector lấy một frame theo mỗi `interval_ms`; có thể đổi sang
`scene-segments` hoặc `linear-rulebase` bằng registry/config. Keyframe có tên theo **số frame nguồn**,
zero-based, sáu chữ số và extension theo profile:

```text
data/L29_a/dataset/keyframes/L21_V030/000000.png
data/L29_a/dataset/keyframes/L21_V030/000025.png
```

Mỗi video có selection manifest, rendered manifest và validation report. Rule
validate kiểm tra file/dung lượng, media probe, checkpoint decode ở các mốc
cấu hình được, mapping selection-render, ảnh đọc được, dimensions, extension
và fingerprint nguồn. Warning FPS bị thiếu không tự làm pipeline fail; lỗi
decode, artifact hoặc media mới làm fail.

### Kaggle staging và cleanup

Kaggle staging là allowlist explicit, thường gồm:

```text
<lot>/kaggle-staging/
├── dataset-metadata.json
├── keyframes/<video_id>/*.png       # mặc định; JPEG vẫn cấu hình được
├── metadata/<video_id>.json
├── manifests/{selection,rendered,validation}/...
├── scene-segments/<video_id>.json   # nếu upload.include_scene_segments=true
├── PECore-features/                    # nếu bật và đã tồn tại
├── transcripts/                        # nếu bật và đã tồn tại
├── keyframe_transcript_index/          # nếu bật và đã tồn tại
└── provenance.json                      # payload/config/runtime inventory
```

Scene segment nguồn được đọc từ `processing.scene_segments_dir` và copy theo
từng video vào `kaggle-staging/scene-segments/`. Khi bật
`upload.include_scene_segments`, manifest của mọi video trong lot phải tồn tại;
thiếu một file sẽ làm staging dừng để tránh upload dataset không đầy đủ. Các
file local trong `data/scene-segments/` vẫn được giữ lại.

`upload.missing_artifact_policy=error` áp dụng cùng nguyên tắc cho features,
transcript và transcript index: thiếu artifact của một video thì staging dừng.
Đặt `skip` nếu các artifact đó là tùy chọn; khi đó chỉ những file hiện có mới
được đưa vào staging.

Staging **không chứa archive, source hoặc video**. Chỉ khi Kaggle CLI upload
thành công, status là `ready` và remote `provenance.json` có payload digest đúng
thì `CleanupManager` mới xóa artifact được bật
trong `cleanup`; metadata độc lập trong `data/metadata/` không bao giờ bị xóa.
Mặc định `upload.staging_scope=lot`: staging nằm ở
`data/<lot_id>/kaggle-staging/` và được cleanup sau khi verify thành công, nên
không phải giữ dữ liệu các lot cũ trên SSH. Dataset remote của mỗi lot vẫn độc
lập và không bị ảnh hưởng.

`upload.staging_scope=dataset` là legacy cumulative mode. Khi bật rõ mode này,
staging nằm ở `data/kaggle-dataset-staging/`, có state riêng ở
`data/kaggle-dataset-state.json` và lock riêng ở
`data/kaggle-dataset-upload.lock`; staging được giữ lại để làm base cho version
Kaggle kế tiếp. Archive, source, keyframe, feature và manifest làm việc của
lot vẫn được cleanup theo config.

Mặc định config mẫu xóa archive, source, keyframe, feature, manifest và các
artifact tạm của lot sau khi verify. Nếu upload tắt, pipeline không tự cleanup.

### Strategy thay thế

`SelectionStrategyRegistry` và `ProcessingStrategyRegistry` cho phép đăng ký
strategy mới theo tên; config chọn `processing.strategy` và `processing.selector`.
`BatchOrchestrator` nhận dependency qua constructor,
vì vậy có thể thay `Aria2ArchiveDownloader`, `KeyframeProcessingStrategy`,
`VideoValidationPipeline`, `KaggleStagingStrategy` hoặc `KaggleCliUploader`
bằng adapter/fake khác mà không sửa workflow chính. `CheckpointStore` lưu
`state.json` và `upload-state.json` cùng event history theo từng lot để SSH có
thể quan sát/resume sau khi một bước thất bại.
