# Preprocess

Các công cụ trong thư mục này xử lý transcript, keyframe và feature visual của
`AIC2026_sample`. Chúng là pipeline offline, không cần chạy FastAPI, và dùng
virtual environment riêng tại `preprocess/.venv`.

## Cấu trúc source

```text
preprocess/
├── README.md
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

Xem chính xác argument của một command:

```bash
preprocess/.venv/bin/python \
  preprocess/transcript/build_keyframe_index.py --help
```

## Dataset layout và quy ước ID

Các script mặc định dùng `<repository-root>/AIC2026_sample`:

```text
AIC2026_sample/
├── transcripts/<video_id>_Transcript.txt
├── metadata/<video_id>.json
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
- Timestamp của frame là `int(int(frame_id) / fps * 1000)`.
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
| `metadata/<video_id>.json` | Lấy `fps`; thiếu thì fallback trong data parser là `25.0`. |
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
preprocess/.venv/bin/python \
  preprocess/transcript/build_keyframe_index.py --force
```

Build một video:

```bash
preprocess/.venv/bin/python \
  preprocess/transcript/build_keyframe_index.py \
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
  "fps": 25.0,
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

Với mỗi frame, `timestamp_ms = int(int(frame_id) / fps * 1000)`. Nếu timestamp
nằm trong `[start_ms, end_ms)` của một sentence, panel chứa `sentence.text`;
nếu không, script vẫn tạo ảnh với panel trống. `anchor_frame_id` không được
dùng bởi renderer này. Với frame có transcript, panel thêm prefix
`Audio content (in Vietnamese): ` trước text sentence.

### Input bắt buộc

| Đường dẫn | Vai trò |
| --- | --- |
| `keyframe_transcript_index/<video_id>.json` | Nguồn `fps`, interval và text sentence. |
| `keyframes/<video_id>/<frame_id>.jpg` | Toàn bộ ảnh source để render. Nếu không có, renderer fallback sang `keyframes_org/<video_id>/`. |

Nếu thư mục keyframe không tồn tại hoặc file ảnh có stem không phải số, script
dừng với lỗi. Không có `continue_on_error`, nhằm phát hiện dữ liệu keyframe bị
lệch ngay lập tức.

### CLI arguments

| Argument | Bắt buộc | Mặc định | Hành vi |
| --- | --- | --- | --- |
| `--video-id <id>` | Có | — | Video cần render. |
| `--sample-root <path>` | Không | `<repo>/AIC2026_sample` | Root dataset thay thế; phải chứa `keyframes/` và `keyframe_transcript_index/`. |
| `--output-dir <path>` | Không | `<sample-root>/subtitled_keyframes` | Root output. Script tạo thêm thư mục `<video_id>/`. |
| `--image-extension <ext>` | Không | `.jpg` | Extension của ảnh anchor, ví dụ `.png`. |
| `--font-path <path>` | Không | Tự tìm DejaVu/Noto/Liberation | Font Unicode `.ttf`/`.otf`. |
| `--font-size <n>` | Không | `30` | Cỡ font pixel; phải dương. |
| `--panel-height <n>` | Không | `200` | Chiều cao panel tối thiểu, pixel; phải dương. |
| `--overflow <mode>` | Không | `error` | `expand`: tăng panel; `truncate`: cắt dòng dư; `error`: dừng nếu panel quá thấp. |
| `--existing <mode>` | Không | `overwrite` | `overwrite`: ghi đè; `skip`: bỏ ảnh đã có; `unique`: thêm `_1`, `_2`, … vào tên file. |
| `--dry-run` | Không | Tắt | Kiểm tra index và danh sách keyframe, không ghi ảnh. |

### Cách chạy

Kiểm tra input không ghi output:

```bash
preprocess/.venv/bin/python \
  preprocess/transcript/add_transcript_banner.py \
  --video-id L03_V002 --dry-run
```

Render với panel tự mở rộng:

```bash
preprocess/.venv/bin/python \
  preprocess/transcript/add_transcript_banner.py \
  --video-id L03_V002 \
  --panel-height 200 \
  --overflow expand
```

Render vào vị trí khác, không ghi đè ảnh cũ:

```bash
preprocess/.venv/bin/python \
  preprocess/transcript/add_transcript_banner.py \
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
preprocess/.venv/bin/python \
  preprocess/extract_keyframes.py \
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
preprocess/.venv/bin/python \
  preprocess/extract_keyframes.py \
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
preprocess/.venv/bin/python \
  preprocess/extract_keyframes.py \
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
python -m pip install -r preprocess/requirements.txt
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
        ├── kaggle-staging/
        ├── reports/
        ├── receipts/
        └── state.json
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
| Frame scanner | Video + `VideoInfo` | `FrameCandidate[]` | Mỗi candidate có `source_frame_number: int`, `timestamp_ms: int`, `pts_time_seconds: float`; `source_frame_number` và PTS lấy từ `showinfo` của chính FFmpeg decoder, không suy ra từ `nb_frames`/FPS. |
| Selection strategy | `VideoInfo` + candidates | `SelectedFrame[]` | `interval_ms`, `start_ms`, `end_ms` dùng milliseconds; số frame là `int`. |
| FFmpeg renderer | Video + selected frames | JPG/PNG + rendered manifest | Mặc định `target_short_edge_px=null`: giữ nguyên kích thước decoded; nếu đặt giá trị thì resize cạnh ngắn; frame ID là chuỗi sáu chữ số. |
| Video validator | VideoAsset + metadata + ProcessingResult | `ValidationReport` | Kiểm tra file, media, decode checkpoint, ảnh, dimensions, mapping và fingerprint; checkpoint là tỷ lệ `0.0..1.0`. |
| `PECoreEmbeddingPipeline` | `keyframes/<video_id>/*.{jpg,jpeg,png}` | `.npy` từng frame | Ảnh RGB được transform vào tensor `(B, 3, 448, 448)` theo PECore; `embedding.batch_size` là số ảnh/inference batch; `embedding.dataloader` điều chỉnh worker/pin/prefetch; encoder trả `(B, 1280)`, `float32`; mỗi file là `(1280,)`, L2 norm gần `1.0`. |
| `KaggleStagingStrategy` | Keyframes, features, metadata, manifests | `kaggle-staging/` | Chỉ là file allowlist; không chứa archive, source hoặc video. |
| `KaggleCliUploader` | `StagingResult` | `UploadResult` | `verified: bool`; upload dạng `create` hoặc `version`. |
| `CleanupManager` | UploadResult đã verify | `CleanupResult` | Xóa các directory được bật trong config; metadata độc lập, reports, receipts và state vẫn giữ lại. |

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
processing.decode_checkpoints          = [0, 0.5, 1] # normalized position
embedding.expected_dim                = 1280 # vector values
embedding.batch_size                  = 8    # images/encoder call
embedding.dataloader.num_workers     = 0    # CPU image preprocessing workers
embedding.dataloader.pin_memory      = false # host→CUDA transfer hint
embedding.dataloader.persistent_workers = false
embedding.dataloader.prefetch_factor = 2    # only when num_workers > 0
progress.enabled                      = true # terminal/tmux progress bars
progress.leave                        = false # clear completed bars; less terminal noise
progress.bar_width                    = 24   # fixed width of each bar
shot_boundary.enabled                 = true # auto-run when selector needs scenes
shot_boundary.backend                 = transnetv2
shot_boundary.device                  = auto # auto/cpu/cuda/mps
shot_boundary.threshold               = 0.5
shot_boundary.overwrite               = false # reuse valid manifests
```

Progress bar dùng `tqdm` và hiển thị đúng năm dòng cố định trong terminal/tmux:
dòng lot/stage, dòng CPU/GPU, dòng disk, full-pipeline bar và detail bar của
operation hiện tại. Các operation lồng nhau dùng lại detail bar nên không tạo
thêm hàng. Cả hai progress bar đều hiển thị `eta`; `bar_width` giới hạn chiều
dài phần bar; `leave=false` xóa hai bar khi pipeline kết thúc.
Khi output đi qua `tee`, pipeline tự chuyển sang một dòng ASCII duy nhất để
không ghi cursor escape code vào log, nhưng vẫn giữ full-pipeline, operation
hiện tại, CPU/GPU/disk metrics và ETA. Để xem layout năm dòng cố định, chạy
trực tiếp trong terminal/tmux không pipe output qua `tee`. Warning CUDA không
gây lỗi pipeline;
TransNetV2 chỉ lọc warning lặp lại về `CUBLAS_WORKSPACE_CONFIG`. Nếu cần tái lập
bit-level, export biến này trước khi khởi động Python theo hướng dẫn của PyTorch.

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
python -m pip install -r preprocess/requirements.txt
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
chạy lại `python -m pip install -r preprocess/requirements.txt`. `aria2c`
không nằm trong Python requirements vì được cài từ APT. Cách quản lý package
bằng APT theo tài liệu Ubuntu và cách cài/auth Kaggle CLI xem
[Ubuntu package management](https://documentation.ubuntu.com/server/how-to/software/package-management/)
và [Kaggle CLI documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md).

Kaggle cần authentication trước khi bật upload. Trên SSH có thể dùng token qua
environment variable, không ghi token vào repository:

```bash
export KAGGLE_API_TOKEN="<your-kaggle-api-token>"
kaggle datasets list
```

Hoặc dùng flow tương tác của CLI:

```bash
kaggle auth login
```

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

Nếu virtual environment không nằm trong `PATH`, thay `python` bằng Python của
environment đó. `preflight` kiểm tra `aria2c`, `ffmpeg`/filter `showinfo`, `ffprobe`, thêm
TransNetV2 khi selector cần scene, thêm `kaggle` khi upload được bật, và kiểm
tra dung lượng trống trước khi tải.

Khi bật `upload.enabled`, tạo file metadata Kaggle tại path cấu hình bởi
`upload.metadata_template` (mặc định trong config mẫu là
`data/kaggle-dataset-metadata.json`) và đặt `upload.dataset_ref` đúng dataset
đích. Không bật upload khi chưa kiểm tra credentials bằng `kaggle datasets list`.

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
git branch --show-current       # phải là preprocess
```

Không push hoặc merge trong workflow này. Mọi thay đổi code vẫn chỉ thuộc
branch `preprocess`.

#### 2. Cài system tools và Python environment

Chạy một lần trên máy mới:

```bash
sudo apt-get update
sudo apt-get install -y aria2 ffmpeg tmux python3 python3-venv python3-pip ca-certificates

python3 -m venv preprocess/.venv
preprocess/.venv/bin/python -m pip install --upgrade pip
preprocess/.venv/bin/python -m pip install -r preprocess/requirements.txt
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
    "overwrite": false
  },
  "processing": {
    "selector": "linear-rulebase",
    "scene_segments_dir": "../../data/scene-segments"
  },
  "embedding": {
    "enabled": true
  },
  "upload": {
    "enabled": true,
    "mode": "version",
    "dataset_ref": "owner/dataset-slug",
    "metadata_template": "../../data/kaggle-dataset-metadata.json"
  }
}
```

Nếu chỉ muốn tạo artifact local, đặt `upload.enabled=false`; khi đó pipeline
không tự cleanup. Nếu bật upload, `dataset-metadata.json` phải tồn tại đúng
đường dẫn và `dataset_ref` phải đúng dataset đích.

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
  run 2>&1 | tee data/logs/batch-run.log
```

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
```

Khi selector cần scene, checkpoint sẽ đi qua `shot_boundaries` rồi
`shot_boundaries_ready` trước `processing`. File boundary được lưu ngoài lot
ở `data/scene-segments/<video_id>.json`; nếu stage bị ngắt, file hợp lệ sẽ
được dùng lại ở lần chạy sau khi `shot_boundary.overwrite=false`.

Mỗi lot còn có `state.json.stages` với các stage `running` hoặc `completed`.
Checkpoint được ghi trước và sau từng stage; nếu SSH bị mất hoặc nhấn
`Ctrl-C`, lần chạy `run` tiếp theo sẽ:

```text
stage đã completed + artifact còn hợp lệ  → restore/skip
stage đang running hoặc artifact hỏng      → chạy lại stage đó
```

Các artifact được kiểm tra lại trước khi skip: ZIP, source video, scene
manifest, discovery report, rendered/validation manifest, `.npy` feature và
upload/cleanup receipt. Fingerprint theo stage và dependency chain cũng cho
phép đổi riêng `embedding.dataloader` mà vẫn giữ lại download, extraction,
TransNetV2 và processing hợp lệ; embedding và các stage phụ thuộc nó sẽ chạy
lại. Không xóa `state.json` để resume.

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
features, manifests và `kaggle-staging/` có thể đã bị xóa theo config. Nếu
upload tắt, các artifact local vẫn được giữ lại.

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
data/L29_a/dataset/keyframes/L21_V030/000022.jpg
  → data/L29_a/dataset/PECore-features/L21_V030/000022.npy
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

Để chạy tự động sau bước validate trong batch pipeline, bật:

```json
{
  "embedding": {
    "enabled": true,
    "model_id": "hf-hub:timm/PE-Core-bigG-14-448",
    "device": "auto",
    "precision": "fp32",
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
├── PECore-features/                    # nếu bật và đã tồn tại
├── transcripts/                        # nếu bật và đã tồn tại
└── keyframe_transcript_index/          # nếu bật và đã tồn tại
```

Staging **không chứa archive, source hoặc video**. Chỉ khi Kaggle CLI upload
thành công và status được verify thì `CleanupManager` mới xóa artifact được bật
trong `cleanup`; metadata độc lập trong `data/metadata/` không bao giờ bị xóa.
Mặc định config mẫu xóa archive, source, keyframe, feature, manifest,
transcript và staging sau khi verify. Nếu upload tắt, pipeline không tự cleanup.

### Strategy thay thế

`SelectionStrategyRegistry` và `ProcessingStrategyRegistry` cho phép đăng ký
strategy mới theo tên; config chọn `processing.strategy` và `processing.selector`.
`BatchOrchestrator` nhận dependency qua constructor,
vì vậy có thể thay `Aria2ArchiveDownloader`, `KeyframeProcessingStrategy`,
`VideoValidationPipeline`, `KaggleStagingStrategy` hoặc `KaggleCliUploader`
bằng adapter/fake khác mà không sửa workflow chính. `CheckpointStore` lưu
`state.json` và event history theo từng lot để SSH có thể quan sát/resume sau
khi một bước thất bại.
