# Handoff hệ thống preprocess

Tài liệu này là điểm bắt đầu cho một coding/SSH session mới. Nó mô tả trạng
thái kiến trúc, dataflow, contract dữ liệu, checkpoint, config và lệnh vận hành
của hệ thống trong `preprocess/`. README là tài liệu tham chiếu chi tiết;
handoff này ưu tiên thứ tự cần đọc và các quyết định vận hành.

## 1. Phạm vi và nguyên tắc làm việc

- Chỉ sửa source trong `preprocess/`. Runtime data nằm trong `data/`.
- Branch phát triển là `preprocess`; không merge vào `main` nếu người dùng chưa
  yêu cầu rõ ràng.
- Chỉ push lên `origin/preprocess` sau khi đã test xong và người dùng yêu cầu.
- Không hardcode đường dẫn máy, Kaggle owner, dataset slug, GPU hoặc credentials.
- Giữ kiến trúc strategy/OOP để downloader, detector, selector, processor,
  embedding, staging và uploader có thể thay thế độc lập.
- Không xóa artifact local trước khi upload được verify bằng status remote và
  `payload_digest` trong `provenance.json`.
- Worktree có thể chứa thay đổi của người dùng, đặc biệt `.gitignore`; không ghi
  đè, reset hoặc commit thay đổi không thuộc task.

Kiểm tra session mới trước khi làm gì:

```bash
cd /path/to/AIC-HCMC-2026
git status --short --branch
git switch preprocess
git pull --ff-only origin preprocess
```

Nếu `git switch` bị chặn vì thay đổi local, dừng và kiểm tra từng file; không
stash/reset tự động.

## 2. Mục tiêu hệ thống

Pipeline nhận một file text có URL ZIP, xử lý tuần tự từng lot và có thể chạy
liên tục trong `tmux`:

```text
links.txt
  -> aria2c download/resume
  -> validate ZIP
  -> unzip video/ và rename thành <lot_id>/
  -> discover + validate video
  -> TransNetV2 scene segments (khi selector cần)
  -> scan timeline PTS + select + render keyframe
  -> validate keyframe/video/manifest
  -> PE-Core embedding (tùy chọn)
  -> dựng Kaggle staging không chứa video
  -> Kaggle create/version
  -> status ready + tải provenance remote + đối chiếu digest
  -> cleanup artifact local theo allowlist
  -> lot tiếp theo
```

Ví dụ identity:

```text
URL/ZIP: https://aic-data.ledo.io.vn/Videos_L29_a.zip
ZIP root: video/
lot_id:  L29_a
source:  data/L29_a/source/L29_a/
video:   L29_V001.mp4
```

Lot được suy ra từ tên ZIP bằng cách bỏ prefix `Videos_` và suffix `.zip`.
Tên video giữ nguyên; pipeline không tự thêm hậu tố lot vào video ID.

## 3. Source map

```text
preprocess/
├── SESSION_HANDOFF.md          # tài liệu này
├── README.md                   # tài liệu đầy đủ + runbook
├── plan.md                     # checklist refactor/verification đã thực hiện
├── requirements.txt            # dependency khai báo
├── requirements.lock.txt       # phiên bản pin cho venv SSH
├── progress.py                 # tqdm, nhiều dòng TTY, CPU/GPU/disk metrics
├── batch/
│   ├── __main__.py             # python -m preprocess.batch
│   ├── cli.py                  # preflight/parse/run/upload/cleanup
│   ├── config.py               # dataclass config + path resolution
│   ├── config.example.json     # nguồn config mẫu
│   ├── orchestrator.py         # stage graph, resume, overlap, locks
│   ├── checkpoints.py          # state atomic + checkpoint stage/video
│   ├── layout.py               # filesystem contract theo lot
│   ├── links.py                # URL -> archive_name + lot_id
│   ├── downloader.py           # aria2c strategy
│   ├── archive_validator.py    # size/path/ZIP bomb/inventory checks
│   ├── archive_extractor.py    # extract tạm + atomic rename
│   ├── video_discovery.py      # source video -> VideoAsset
│   ├── shot_boundaries.py      # TransNetV2 + per-video JSON cache
│   ├── processor.py            # registries + selection strategies
│   ├── validators.py           # video/keyframe/checkpoint validation
│   ├── embedding.py            # PE-Core batch strategy adapter
│   ├── kaggle_uploader.py      # staging, create/version, verify remote
│   ├── cleanup.py              # verified-only, lot-owned deletion
│   ├── locks.py                # run/upload/dataset file locks
│   ├── provenance.py           # SHA-256, runtime inventory, atomic JSON
│   └── dataset_state.py        # cumulative staging legacy
├── keyframes/
│   ├── contracts.py            # VideoInfo, FrameRef, RenderProfile...
│   ├── selectors/              # uniform, scene-segments, linear-rulebase
│   └── extractors/ffmpeg.py    # probe, PTS scan, decoder mapping, render
├── pecore/embedding.py         # model, precision/autocast/TF32, DataLoader
├── transcript/                 # transcript index/banner utilities
└── tests/                      # unit/integration-style local tests
```

## 4. Runtime layout

Chỉ cần `preprocess/` và `data/` trên SSH. Với config đặt tại
`preprocess/batch/config.json`, path `../../data` được resolve tương đối với
thư mục chứa config, tức repository `data/`.

```text
data/
├── input/
│   └── links.txt
├── metadata/
│   └── <video_id>.json
├── scene-segments/
│   └── <video_id>.json
├── kaggle-dataset-metadata.json
├── logs/
└── <lot_id>/
    ├── archive/
    │   └── Videos_<lot_id>.zip
    ├── source/
    │   └── <lot_id>/<video_id>.mp4
    ├── dataset/
    │   ├── keyframes/<video_id>/<frame_id>.png
    │   ├── keyframes/<video_id>/manifest.json
    │   ├── selection-manifests/<video_id>.json
    │   ├── PECore-features/<video_id>/<frame_id>.npy
    │   ├── PECore-features/<video_id>/provenance.json
    │   ├── transcripts/                         # nếu có
    │   └── keyframe_transcript_index/           # nếu có
    ├── kaggle-staging/                          # snapshot riêng lot
    │   ├── dataset-metadata.json
    │   ├── keyframes/
    │   ├── PECore-features/
    │   ├── metadata/
    │   ├── manifests/
    │   ├── scene-segments/                      # nếu include=true
    │   └── provenance.json
    ├── reports/
    │   ├── decoder-timelines/<video_id>.json
    │   ├── embedding-completions/<video_id>.json
    │   ├── validation/<video_id>.json
    │   ├── embedding.json
    │   ├── staging.json
    │   └── upload-disk-budget.json
    ├── receipts/
    │   ├── upload.json
    │   └── cleanup.json
    ├── state.json
    ├── upload-state.json
    ├── run.lock
    └── upload.lock
```

`staging_scope=dataset` là chế độ cumulative legacy và dùng thêm
`data/kaggle-dataset-staging/`. Mặc định/khuyến nghị hiện tại là
`staging_scope=lot`: mỗi lot là một Kaggle dataset riêng, sau verify có thể xóa
toàn bộ payload local của lot mà không cần giữ lot cũ.

## 5. Contract từng stage

| Stage | Input chính | Output chính | Resume/validate |
|---|---|---|---|
| `download` | `ArchiveInput(url, archive_name, lot_id)` | ZIP trong `archive/` | aria2c `--continue=true`, kiểm tra file và `.aria2` |
| `archive_validate` | ZIP | inventory, size, SHA-256 | chống traversal/ZIP bomb, root phải là `video/` |
| `extract` | ZIP đã validate | `source/<lot_id>/` | extract vào temp rồi atomic rename `video -> lot_id` |
| `discover` | source root | danh sách `VideoAsset` | tên mặc định, extension allowlist |
| `shot_boundaries` | từng video | `scene-segments/<video_id>.json` | checkpoint/cache từng video; TransNetV2 khi cần |
| `process_validate` | video + selector/scene JSON | keyframe + selection/render/validation manifests | checkpoint từng video, xác minh source/config/hash |
| `embedding` | keyframe RGB | `.npy` `(1280,)` float32 + provenance | checkpoint từng video; batch inference cấu hình được |
| `stage_upload` | artifact đã validate | `kaggle-staging/`, upload receipt | staging allowlist, digest toàn payload, remote verify |
| `cleanup` | upload verified | cleanup receipt | chỉ xóa target lot-owned đã bật trong config |

Thứ tự stage được tạo động. Nếu tắt detector, embedding hoặc upload thì stage
tương ứng không có trên progress/state.

### Input/output tensor và media

- Video input: file media, pipeline probe bằng `ffprobe`; FPS có thể `null`.
- Timeline selection dùng PTS decoder và frame presentation order, không suy
  timestamp từ FPS. Điều này hỗ trợ VFR và video thiếu FPS metadata.
- `FrameRef.source_frame_number` zero-based; `frame_id` là sáu chữ số, ví dụ
  frame 3084 thành `003084.png`.
- Mặc định `target_short_edge_px=null`, `image_format=png`: giữ nguyên kích
  thước frame decoded và không upscale.
- PE-Core transform ảnh RGB thành tensor `(B, 3, 448, 448)`; output model được
  validate thành vector normalized `(1280,)`, lưu `.npy` float32 từng frame.
- `embedding.batch_size` là số ảnh mỗi inference batch, không phải video/lot.
- `scheduling.max_pending_embeddings` là số video render xong tối đa đang chờ
  một embedding worker; không tạo nhiều model/GPU worker.

## 6. Selection strategies

`processing.strategy` hiện có `keyframes`. Policy chọn frame nằm trong
`processing.selector`:

1. `uniform`
   - Không cần scene boundary.
   - Chọn frame đầu tiên tại/sau mỗi mốc `interval_ms` theo PTS.

2. `scene-segments`
   - Cần per-video scene JSON.
   - Shot `<=3s`: 1 frame; `<=10s`: 3 frame; dài hơn: 5 frame.
   - Quy tắc này hiện nằm trong implementation strategy.

3. `linear-rulebase` (alias `linear`)
   - Cần per-video scene JSON.
   - Toàn bộ threshold/count đọc từ `processing.linear_rule`.
   - Config hiện mô tả: `<=1s -> 1`, `>1s và <=3s -> 2`, sau 3 giây cứ mỗi
     3 giây cộng 1 frame. Frame đặt gần tâm các bucket thời gian trong shot.

Với selector cần scene, nếu `shot_boundary.enabled=true`, config loader tự đặt
cả detector output và processor input về `data_root/scene-segments` khi path là
`null`. Nếu khai báo cả `shot_boundary.output_dir` và
`processing.scene_segments_dir`, hai path phải giống nhau.

## 7. Checkpoint, resume và concurrency

### Main state

`data/<lot_id>/state.json` chứa request, config fingerprint, events, stage
fingerprints và checkpoint từng video. Trạng thái `running` ở cấp stage không
có nghĩa mọi video đã hoàn thành; xem:

```text
stages.<stage>.status
stages.<stage>.videos.<video_id>.status
current_stage
current_video
```

Khi chạy lại cùng request/config:

- Stage hoàn tất có fingerprint hợp lệ được restore.
- Trong `shot_boundaries`, `process_validate` và `embedding`, video đã hoàn tất
  và artifact còn hợp lệ được restore; chỉ video thiếu/stale được chạy lại.
- Partial keyframe directory của đúng video/profile được xóa và dựng lại; video
  khác không bị ảnh hưởng.
- Lot `completed` hợp lệ được tự skip trong `run`.
- Đổi config ảnh hưởng stage nào sẽ đổi fingerprint stage đó và downstream cần
  validate/rerun; artifact cũ không được xóa mù quáng.

### Upload state

`upload-state.json` tách khỏi `state.json`, vì vậy có thể chạy upload riêng mà
không tranh ghi checkpoint xử lý. `upload --lot-id` lấy lock toàn lot và lock
upload; không chạy đồng thời với `run` trên cùng lot.

### Overlap

- `overlap_render_embedding=true`: một GPU embedding worker nhận video ngay sau
  khi video render+validate xong; model PE-Core load lazy ở batch đầu. Queue
  tối đa `max_pending_embeddings` video và backpressure render nếu đầy. Video
  không bị bỏ qua khi render nhanh hơn embed.
- Sau interrupt/code update, completion journal và per-video checkpoint giúp
  khôi phục video render trước đó rồi enqueue/embed video chưa có feature hợp lệ.
- `overlap_upload=true`: upload lot trước bằng một worker trong khi foreground
  xử lý lot kế tiếp. Chỉ một pending upload được phép để giới hạn disk.
- Không chạy song song lệnh `upload --lot-id` với `run` cho cùng lot.

## 8. Progress UI

Trong TTY/tmux, `TqdmProgressReporter` giữ vùng nhiều dòng và refresh tại chỗ:

1. lot/stage hiện tại;
2. CPU/GPU/device/VRAM;
3. disk used/free;
4. full pipeline elapsed/ETA/rate;
5. detail scan/render/video;
6. embedding activity;
7. upload activity.

Chạy trực tiếp để thấy giao diện nhiều dòng. Khi pipe qua `tee`, output không
còn là TTY nên reporter chuyển sang một dòng carriage-return; log dễ lưu nhưng
không giữ giao diện cố định như tmux.

## 9. Config cần biết

Luôn bắt đầu từ `preprocess/batch/config.example.json`, không dùng một config cũ
thiếu field mới. Các path tương đối được resolve theo vị trí file config.

### Cấu hình khuyến nghị cho GPU CUDA hiện đại

Không hardcode model GPU. Bắt đầu an toàn bằng `device=auto`, đo VRAM rồi tăng
batch/window:

```json
{
  "shot_boundary": {
    "enabled": true,
    "backend": "transnetv2",
    "device": "auto",
    "threshold": 0.5,
    "window_batch_size": 1,
    "overwrite": false
  },
  "processing": {
    "strategy": "keyframes",
    "selector": "linear-rulebase",
    "scene_segments_dir": "../../data/scene-segments",
    "profile_id": "keyframes",
    "target_short_edge_px": null,
    "image_format": "png",
    "png_compress_level": 6,
    "ffmpeg_threads": 0,
    "overwrite": false
  },
  "embedding": {
    "enabled": true,
    "model_id": "hf-hub:timm/PE-Core-bigG-14-448",
    "device": "auto",
    "precision": "fp32",
    "autocast": {"enabled": true, "dtype": "bf16", "cache_enabled": true},
    "tf32": {"enabled": true, "matmul": true, "cudnn": true},
    "expected_dim": 1280,
    "batch_size": 16,
    "dataloader": {
      "num_workers": 2,
      "pin_memory": true,
      "persistent_workers": false,
      "prefetch_factor": 2
    }
  },
  "scheduling": {
    "overlap_render_embedding": true,
    "max_pending_embeddings": 10,
    "overlap_upload": false,
    "max_pending_uploads": 1
  }
}
```

`precision` điều khiển dtype model. `autocast` điều khiển mixed precision cho
inference. TF32 chỉ tăng tốc phép toán float32 CUDA được hỗ trợ; nó không đổi
model sang bf16 và không thay thế autocast. Request precision chỉ bị từ chối
khi backend/device thực tế không hỗ trợ.

### Reproducibility

- `best_effort`: ghi seed/config/runtime/tool/model provenance, ưu tiên chạy được
  trên phần cứng khác nhau.
- `strict`: bật deterministic policy và yêu cầu `model_revision` là commit SHA
  Hugging Face 40 ký tự khi model dùng `hf-hub:`.
- Muốn deterministic CuBLAS, biến môi trường phải tồn tại trước khi Python
  khởi động:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Ngay cả khi pin package, bitwise equality giữa GPU/driver/FFmpeg khác nhau
không được đảm bảo; `provenance.json` ghi revision, package/tool và device để
traceback.

## 10. Kaggle model và upload contract

Khuyến nghị mỗi lot một dataset:

```json
{
  "upload": {
    "enabled": false,
    "mode": "auto",
    "dir_mode": "zip",
    "staging_scope": "lot",
    "dataset_ref_template": "OWNER/aic2026-hcmc-{lot_slug}",
    "metadata_template": "../../data/kaggle-dataset-metadata.json",
    "include_scene_segments": true,
    "include_features": true,
    "include_transcripts": true,
    "include_transcript_index": true,
    "require_remote_inventory": true,
    "missing_artifact_policy": "error",
    "temporary_space_multiplier": 1.0,
    "public": false
  }
}
```

- `enabled=false`: full `run` chỉ tạo local artifact. Lệnh explicit
  `upload --lot-id` vẫn override flag này và upload.
- `mode=auto`: `datasets status`; dataset tồn tại -> `version`, chưa tồn tại ->
  `create`.
- `dataset_ref_template`: `L29_a` + `{lot_slug}` -> `l29-a`.
- Chỉ cấu hình một trong `dataset_ref` và `dataset_ref_template`.
- `dir_mode=zip`: Kaggle CLI zip từng top-level directory. Phải dự trù disk tạm;
  PNG/NPY thường nén thêm không nhiều.
- Staging luôn chứa keyframe, metadata và manifests; không chứa ZIP/source/video.
- Features/scene/transcript được kiểm soát bằng các cờ include.

Metadata template local:

```json
{
  "title": "AIC2026-HCMC",
  "id": "OWNER/placeholder-dataset",
  "licenses": [{"name": "other"}]
}
```

Pipeline ghi đè `id` trong bản staging bằng target thực tế; file nguồn phải tên
gì cũng được nhưng path `upload.metadata_template` phải trỏ đúng. Bản staging
luôn tên `dataset-metadata.json`.

Upload chỉ thành công khi:

1. Kaggle CLI transfer trả exit 0;
2. dataset status thành `ready`;
3. pipeline tải đúng remote `provenance.json`;
4. remote `payload_digest` bằng local digest;
5. receipt `receipts/upload.json` được ghi `verified=true`.

Sau đó cleanup mới chạy. Thanh transfer 100% chỉ có nghĩa blob/part hiện tại đã
truyền xong; vẫn còn create/version, status poll, provenance verify và cleanup.

## 11. Setup Ubuntu/SSH từ đầu

### System tools

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv aria2 ffmpeg tmux git
aria2c --version
ffmpeg -version
ffprobe -version
```

CUDA driver/PyTorch phải phù hợp máy thuê; không cài driver hệ thống một cách
mù quáng từ requirements Python.

### Venv riêng

```bash
cd /path/to/AIC-HCMC-2026
python3 -m venv preprocess/.venv
source preprocess/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r preprocess/requirements.lock.txt
python -m pip check
```

Lock hỗ trợ Python >=3.10; marker chọn Kaggle CLI phù hợp Python. Kiểm tra:

```bash
python --version
python -m pip show torch transnetv2-pytorch open_clip_torch kaggle
python -c 'import torch; print("cuda=", torch.cuda.is_available()); print("device=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")'
which python
which kaggle
```

### Input

```bash
mkdir -p data/input data/metadata data/logs
cp preprocess/batch/config.example.json preprocess/batch/config.json
```

`data/input/links.txt` có một URL mỗi dòng; dòng trống và dòng bắt đầu `#` bị
bỏ qua:

```text
https://aic-data.ledo.io.vn/Videos_L29_a.zip
https://aic-data.ledo.io.vn/Videos_L27_a.zip
```

Metadata độc lập với ZIP:

```text
data/metadata/L29_V001.json
data/metadata/L29_V002.json
```

### Kaggle authentication

Không lưu secret trong config/repository. Với legacy key:

```bash
mkdir -p /root/.kaggle
# chép kaggle.json tải từ Kaggle Settings/API vào đây
chmod 600 /root/.kaggle/kaggle.json
kaggle datasets list --mine --page-size 5
```

Việc xem dataset public không chứng minh đã authenticated; `list --mine` là
kiểm tra bắt buộc. Kiểm tra không in secret:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path.home() / ".kaggle" / "kaggle.json"
d = json.loads(p.read_text())
print("username=", d.get("username"))
print("key_present=", bool(d.get("key")))
print("permissions=", oct(p.stat().st_mode & 0o777))
PY
```

`which kaggle` phải cùng venv với Python pipeline hoặc `tools.kaggle` phải là
đường dẫn tuyệt đối tới executable mong muốn.

## 12. Workflow tmux hoàn chỉnh

```bash
ssh USER@HOST
cd /path/to/AIC-HCMC-2026
git switch preprocess
git pull --ff-only origin preprocess
source preprocess/.venv/bin/activate
tmux new -s aic-preprocess
```

Trong tmux:

```bash
cd /path/to/AIC-HCMC-2026
source preprocess/.venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -m preprocess.batch \
  --config preprocess/batch/config.json \
  preflight

python -m preprocess.batch \
  --config preprocess/batch/config.json \
  parse-links

python -m preprocess.batch \
  --config preprocess/batch/config.json \
  run --continue-on-error
```

Detach bằng `Ctrl-b`, sau đó `d`. Reattach:

```bash
tmux attach -t aic-preprocess
```

Không cần `tee`; tmux giữ tiến trình và giao diện progress đẹp hơn. Nếu cần log
thô:

```bash
set -o pipefail
python -m preprocess.batch \
  --config preprocess/batch/config.json \
  run --continue-on-error 2>&1 | tee data/logs/batch-run.log
```

### Upload/cleanup riêng

```bash
python -m preprocess.batch \
  --config preprocess/batch/config.json \
  upload --lot-id L29_a

python -m preprocess.batch \
  --config preprocess/batch/config.json \
  cleanup --lot-id L29_a
```

Lệnh upload explicit không cần `upload.enabled=true`, nhưng toàn bộ field target,
metadata, include và verify vẫn phải hợp lệ. Cleanup riêng reverify remote trước
khi xóa.

## 13. Chẩn đoán nhanh

### Path/config

```bash
python -c 'from pathlib import Path; from preprocess.batch.config import BatchConfig; c=BatchConfig.from_json(Path("preprocess/batch/config.json")); print(c.data_root); print(c.links_file); print(c.metadata_root); print(c.processing.scene_segments_dir); print(c.upload.metadata_template); print(c.upload.target_for_lot("L29_a"))'
```

### Checkpoint lot

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("data/L29_a/state.json")
s = json.loads(p.read_text())
print("state=", s.get("state"))
print("current_stage=", s.get("current_stage"))
print("current_video=", s.get("current_video"))
for name, stage in s.get("stages", {}).items():
    videos = stage.get("videos", {})
    done = sum(v.get("status") == "completed" for v in videos.values())
    print(name, stage.get("status"), f"videos={done}/{len(videos)}")
PY
```

### Kaggle

```bash
chmod 600 /root/.kaggle/kaggle.json
which kaggle
kaggle --version
kaggle datasets list --mine --page-size 5
kaggle datasets status OWNER/DATASET
```

Chỉ retry upload khi `datasets list --mine` xác nhận authentication. Không xóa
`/tmp/.kaggle/uploads` trong quá trình retry vì Kaggle CLI dùng nó cho resumable
upload. Thanh transfer 100% không thay thế bước create/version và remote verify.

### Media

- `FFmpeg decoded ... but selector requested ...` hoặc `timeline cannot map`:
  source có chênh lệch ffprobe/decoder/VFR. Code hiện có fallback map PTS và
  checkpoint theo video; chạy lại cùng config để retry đúng video.
- Cảnh báo CuBLAS deterministic không phải lỗi; strict run nên export
  `CUBLAS_WORKSPACE_CONFIG` trước Python.
- TransNetV2 chậm: kiểm tra progress device, `torch.cuda.is_available()`,
  `nvidia-smi`, `shot_boundary.device` và `window_batch_size`.

## 14. Verification trước commit/push

Chạy từ repository root với venv preprocess:

```bash
source preprocess/.venv/bin/activate
python -m unittest discover -s preprocess/tests -v
python -m compileall -q preprocess
python -m pip check
python -m preprocess.batch --help
python -m preprocess.batch \
  --config preprocess/batch/config.example.json \
  parse-links --help
git diff --check -- preprocess
git status --short
```

Không đưa `data/`, credentials, `.venv/`, cache hoặc thay đổi `.gitignore` của
người dùng vào commit. Commit chỉ các file `preprocess/` thuộc task, rồi push:

```bash
git push origin preprocess
```
