# Keyframe pipeline experiments — L01_V002.mp4

## Mục tiêu

So sánh các kiến trúc xử lý video (TransNetV2 shot boundary → chọn keyframe → PE-Core embed) giữa
local machine (CPU mạnh, không GPU) và Colab (GPU T4, CPU yếu — free tier ~2 vCPU), để tìm kiến
trúc nhanh nhất mà không đổi chất lượng.

## Setup

- **Colab**: `colab attach -s gpu` + `colab config-ssh -s gpu` mỗi lần mở session mới (session
  ephemeral, mất hết `/content` khi crash/restart — đã xảy ra nhiều lần trong quá trình test).
  SSH chỉ cho 1 connection cùng lúc.
- **Local**: cài `ffmpeg` qua `scoop install ffmpeg` (bản 9.0) — ban đầu máy không có ffmpeg binary.
- **Repo**: tìm thấy branch `origin/preprocess` (chưa merge) có sẵn module `preprocess/` khá hoàn
  chỉnh (keyframe selection đúng `docs/keyframe_selection.md`, PE-Core embedding) nhưng thiết kế
  cho compute tại chỗ, không có cơ chế offload GPU qua Colab.

## Kết quả các thí nghiệm (video 28498 frame, 274 scene tham chiếu, 586 keyframe theo ladder)

| # | Thí nghiệm | Thời gian | Chất lượng |
|---|---|---|---|
| 0 | Baseline gốc (local decode+dedup+transfer+Colab infer) | 287s | 274/274 bit-exact |
| 1 | `predict_video()` built-in (ffmpeg decode qua thư viện, toàn Colab) | ~290s | 273/274 |
| 2 | `preprocess/` module chạy trên Colab (probe+scan+materialize+embed, 586 keyframe thật) | 1422s | — (embed chiếm 67%) |
| 3 | **Merged Pass 1**: gộp 2 lượt decode (scan timestamp + TransNetV2 48×27) thành 1 lệnh ffmpeg, chạy trên Colab | **207.28s** (183.21 decode + 24.07 infer) | 273/274 (lệch 1 frame cuối) |
| 4 | **Merged Pass 2**: pipe full-res frame thẳng vào PE-Core (không qua JPEG đĩa), overlap decode/embed, trên Colab | **891.75s** (88s load + 803.73s overlap) | 586/586, dim=1280, norm≈1.0 |
| 5 | Local full-res extract (586 frame, đã fix bug) | **21.62s** | 586/586 file hợp lệ |
| 6 | Local merged decode (48×27+timestamp) + transfer 107MB + Colab GPU infer thuần | **134.78s** (76.97+33.17+24.64) | 273/274 (nhiều điểm lệch hơn #3, do khác version ffmpeg) |

**So với baseline 287s**: merged Pass 1 nhanh hơn 28%; kịch bản local-decode+Colab-infer (#6) nhanh hơn 53%.

## Bug tìm thấy & sửa

1. **ffmpeg 9.0 (local) giới hạn ~100-200 điều kiện `eq(n,X)` nối nhau** trong filter `select` (ffmpeg
   4.4.2 trên Colab không bị) → sửa bằng `-filter_complex split` thành nhiều nhánh ≤90 điều kiện,
   vẫn giữ đúng 1 lần decode.
2. **`-fps_mode` (ffmpeg mới) vs `-vsync` (ffmpeg cũ)** — 2 máy dùng version khác nhau nên option
   khác tên, phải chọn đúng theo từng máy.
3. **Bug nghiêm trọng nhất**: đặt `-map 0:v:0` đứng trước `-filter_complex` khiến nó bị cộng dồn vào
   output kế tiếp — 1 output nhận nhầm cả 2 luồng (gốc full-res + đã lọc), gây pipe phải cõng
   ~177GB dữ liệu thừa. Hậu quả: crash RAM Colab (gần OOM ở 9.9GB) và ghi ra 33GB rác trên ổ đĩa
   local (29084 file thay vì 586). Tìm ra bằng cách test nhỏ (giới hạn `-frames:v`, so byte thực tế
   với kỳ vọng) trước khi tin tưởng chạy full — từ đó thêm rule vào `CLAUDE.md`.
4. `subprocess.run(capture_output=True)` buffer không giới hạn — đổi sang đọc streaming qua
   `Popen` + cap kích thước an toàn + thread riêng đọc `stderr` (tránh deadlock).

## Chất lượng: vì sao không bao giờ bit-exact giữa các phương pháp decode khác nhau?

Mọi lần đổi cách decode (cv2 ↔ ffmpeg, hoặc ffmpeg version khác nhau) đều cho 273-274/274 scene,
lệch 1-4 điểm ở các cut yếu gần ngưỡng 0.5 — do sai số làm tròn màu (YUV→RGB) khác nhau giữa các
decoder. Không phải bug, là đặc tính vốn có. Nếu cần khớp tuyệt đối với reference gốc (cv2-based),
nên giữ decode trên Colab (ffmpeg 4.4.2, khớp gần nhất). Nếu chấp nhận ~99.6% khớp thì decode ở đâu
cũng được.

## Task 4: sweep batch_size — kết luận: KHÔNG giúp gì (đã đo xong)

| batch_size | Thời gian (586 ảnh) | img/s | peak VRAM |
|---|---|---|---|
| 8 | 815.4s | 0.72 | 10.55GB |
| 16 | 827.7s | 0.71 | 11.31GB |
| 24 | 828.9s | 0.71 | 12.06GB |
| 32 | 827.1s | 0.71 | 12.82GB |

Throughput **phẳng tuyệt đối** dù tăng batch — không OOM, VRAM tăng tuyến tính đúng như dự kiến
(hoạt động bình thường), chỉ là không nhanh hơn.

**Đã cô lập nguyên nhân bằng benchmark riêng** (`bench_pure_gpu_compute.py`): preload sẵn toàn bộ
586 tensor lên GPU trước (17.5s, không tính giờ), rồi CHỈ đo thời gian `model.encode_image()` thuần,
không còn CPU preprocess trong vòng lặp:
- batch_size=8 (pure GPU): **813.3s, 0.72 img/s** — gần như y hệt bản gốc có CPU (815.4s). CPU
  preprocess chỉ chiếm ~2% tổng thời gian → **không phải nút thắt**.
- batch_size=24 (pure GPU): **đang chạy dở khi hết session** (mốc 120/586 ảnh @ 133.0s, 0.90 img/s
  — nhanh hơn batch=8 ở early-batch, cần số cuối để kết luận chắc, xem "Việc dang dở" bên dưới).

**Kết luận**: PE-Core-bigG-14-448 (fp32) trên T4 là **compute-bound thuần** — không phải do thiếu
overlap CPU/GPU, không phải do batch nhỏ. Tăng `batch_size`/`num_workers` sẽ không giúp gì thêm.

## Ý tưởng đã bàn nhưng chưa test xong (để mai làm tiếp)

1. **`torch.compile(model)`** — kernel fusion, cùng phép toán fp32, có thể được 10-30%. Có warmup
   cost (có thể vài chục giây - vài phút cho model 2B tham số), chỉ đáng nếu chạy nhiều batch sau đó
   (pipeline thật thì đáng, test 1 lần thì chưa chắc). **Chưa viết script, chưa test.**

2. **Vision-only checkpoint** (`hf_hub:timm/vit_pe_core_gigantic_patch14_448.fb`) — bản HuggingFace
   chỉ chứa vision encoder (1.88B tham số, không có text tower), load qua `timm` trực tiếp thay vì
   `open_clip`. Việc tách visual-only KHÔNG tăng tốc embed (vì `encode_image()` chưa bao giờ chạy qua
   text tower), chỉ tiết kiệm VRAM + thời gian load model. **CONFIRMED** (xem mục "Xác minh
   vision-only checkpoint" bên dưới) — khớp tuyệt đối với `model.visual` của full checkpoint cả ở
   mức weight (sha256 hash) lẫn mức inference (cosine=1.0), đã áp dụng vào `bench_fp16_vs_fp32.py`.

3. **Packet-pipeline transfer** (chia nhiều ảnh thành N packet tar, vừa transfer packet sau vừa xử
   lý packet trước) — đã viết và validate round-trip ở local (`pack_into_packets.py`,
   `colab_packet_watcher.py`) nhưng **chưa test thật trên Colab** (do giới hạn 1 SSH connection, xem
   mục dưới). Có giá trị cho task 5 (nhiều video), ít giá trị cho lần chạy 1 lần.
   - Phát hiện quan trọng đi kèm: `scp -r` trên 586 file nhỏ mất **829.6s** cho 217MB, trong khi 1
     file lớn 344MB chỉ mất 66-98s — chậm hơn ~13-20 lần do overhead per-file qua SSH proxy Colab.
     **Luôn gộp nhiều file nhỏ thành 1 archive trước khi transfer.**

## Bài học về SSH/Colab session (quan trọng, áp dụng mọi lần sau)

- **Chỉ 1 SSH connection cùng lúc tới 1 Colab runtime (per account)** — mọi lệnh dài đều giữ
  connection tới khi xong hẳn, không mở được connection thứ 2 để check tiến độ/kill song song.
- **`nohup` KHÔNG tách tiến trình khỏi session/terminal** — chỉ chặn SIGHUP, connection vẫn bị giữ.
- **`setsid` đơn thuần cũng không đáng tin cậy** — chỉ work khi là lệnh đầu tiên của job nền
  (`ssh host "setsid cmd < /dev/null > log 2>&1 & echo done; sleep 1"`, xác nhận trả về ~2-4s), nhưng
  **fail âm thầm** khi đứng sau `cd dir &&` (nghi do chính process gọi `setsid` đã là process-group
  leader của subshell nền, khiến syscall `setsid()` trả EPERM). **Cách đúng: luôn dùng `setsid -f`**
  (force fork), đã test và xác nhận hoạt động đúng cho mọi cấu trúc lệnh (kể cả có `export ...; cd
  dir &&` phía trước) — xem `CLAUDE.md` để có pattern chuẩn.
- **`TaskStop` (dừng task nền local) không kill được tiến trình trên Colab** — chỉ ngắt client SSH
  phía tôi, để lại tiến trình mồ côi trên Colab, và có thể để lại "connection ma" (proxy báo
  "already-active" dù không có gì thật đang chạy) — cần đợi tự hết hoặc nhờ user `!kill -9 <pid>` qua
  cell notebook riêng (kênh độc lập, không qua giới hạn 1-connection).
- **Không dùng `> log 2>&1` một chiều nếu muốn xem tiến độ live** — dùng `... 2>&1 | tee log`.
- **Token proxy SSH tự hết hạn sau ~1 tiếng** dù runtime vẫn khoẻ — `colab attach -s X` +
  `colab config-ssh -s X` lại là đủ (kiểm tra `/content` còn nguyên = chỉ token hết hạn, không phải
  crash thật).
- **`scp -r` trên nhiều file nhỏ chậm hơn ~13-90x so với 1 file archive** (829.6s cho 586 file/217MB
  vs 8.9s cho 1 tar 226MB cùng nội dung) — **luôn `tar` lại trước khi transfer nhiều file**.
- **Đa tài khoản Colab chạy song song thật sự (phát hiện mới, rất giá trị)**: `colab --profile <tên>
  attach -s <session>` + `colab --profile <tên> config-ssh -s <session>` cho phép nhiều session ở
  nhiều account Google khác nhau cùng hoạt động — mỗi session vẫn giới hạn 1 connection riêng, nhưng
  vì là 2 account/2 runtime khác nhau nên **không giới hạn lẫn nhau** → 2 (hoặc hơn) GPU T4 free tier
  chạy song song thật. `--profile` là global flag, đặt TRƯỚC subcommand. Lưu ý: `config-ssh` mỗi lần
  gọi **ghi đè toàn bộ block quản lý** (chỉ giữ 1 session/lần) — muốn giữ nhiều host cùng lúc trong
  `~/.ssh/config` phải tự thêm entry thủ công cho các session khác, đặt NGOÀI vùng
  `# BEGIN/END colab-cli managed hosts` để không bị ghi đè ở lần gọi sau.
- Mọi script chạy >30s phải in progress mỗi batch (nếu mỗi batch >2-3s) với `flush=True` — rule đã
  cập nhật trong `CLAUDE.md`.

## Optimize-Plan (kernel Triton "Emulated FP32") — kết luận: DỪNG, đã có dữ liệu thật

User đề xuất kế hoạch (từ 1 AI khác) viết custom Triton kernel giả lập độ chính xác FP32 bằng cách
tách toán hạng FP32 thành 2 phần FP16 (hi/lo), nhân qua Tensor Core rồi cộng dồn ở FP32 — kỹ thuật có
thật trong HPC ("3xTF32"/"Ozaki scheme"), nhưng độ phức tạp không tương xứng với timeline cuộc thi.
Thực hiện theo hướng staged, rẻ trước đắt sau:

- **Stage 0 — `setsid -f`**: xác nhận hoạt động, xem mục SSH ở trên.
- **Stage 2a — de-risk bằng PyTorch thuần (không viết kernel)**: `torch.matmul(fp16,fp16)` luôn trả
  **output fp16**, làm mất hết độ chính xác fp32 dù Tensor Core có tích luỹ nội bộ ở fp32 — kết quả
  bản "emulated" (3 lệnh matmul cộng dồn) cho sai số **6-8.5e-02**, gần bằng fp16 thuần không emulate
  gì (8.29e-02) → chứng minh test kiểu PyTorch thuần **không đủ khả năng** đánh giá đúng ý tưởng, bắt
  buộc phải viết kernel Triton thật (fused, accumulator ở SRAM) mới có câu trả lời thật.
- **Stage 2b — Triton kernel thật** (`emulated_matmul_kernel`, chuẩn block-tiled GEMM, accumulator
  `tl.float32` giữ suốt K-loop): 
  - Đúng: shape nhỏ (128×128×128) sai số **3.4-3.8e-05** (tốt hơn PyTorch thuần ~1000-2000 lần,
    xác nhận cách làm đúng hướng) nhưng vẫn cách ngưỡng `<1e-5` ~3-4 lần — test thêm cả bản 3-term và
    4-term (thêm lại số hạng Xl@Wl đã bỏ) cho kết quả gần như y hệt nhau → sai số này là **floor cố
    hữu** (thứ tự cộng dồn floating-point khác cuBLAS), không phải bug sửa được bằng thêm/bớt số hạng.
  - Ở shape thật (8192×1536×1536, đúng kích thước PE-Core): sai số tăng lên **7.6e-04** (~76 lần
    ngưỡng) — tăng theo K như dự đoán.
  - **Tốc độ: kernel chậm hơn fp32 thuần 178 LẦN** (1666ms/call vs 9.4ms/call, so với mục tiêu lý
    thuyết ~2.7x NHANH hơn) — nghi do BLOCK_K=32 quá nhỏ khiến K-loop bị unroll thành 192 lệnh
    `tl.dot` tuần tự, gây register spilling/occupancy cực thấp, chưa được autotune cho T4.
  - **User quyết định: dừng ở đây** — cả đúng lẫn nhanh đều fail rõ ràng ở bản đầu tiên; sửa được cần
    tuning Triton chuyên sâu (autotune block size, num_warps/num_stages), không tương xứng effort.
- **Stage 3 (EmulatedLinear + full-model)**: huỷ, gated on Stage 2 mà Stage 2 đã fail.

## Isolation test: kernel Triton "chết" vì lý do gì — Tensor Core hay logic emulate?

Trước khi từ bỏ hẳn Triton, chạy 1 kernel tối giản (`test_minimal_triton_matmul.py`, chỉ 1×`tl.dot`,
không tách hi/lo) để cô lập nguyên nhân của mức chậm 178 lần ở Stage 2b:

- Kernel tối giản **cũng chậm bất thường**: **23.99 lần chậm hơn cuBLAS fp16** (34.4ms vs 1.4ms),
  chậm hơn cả fp32 thuần (0.27x). → chứng minh vấn đề KHÔNG nằm ở logic emulate 4-term, mà ở cấu hình
  Triton/grid chung trên T4.
- Debug sâu hơn (`test_triton_micro_debug.py`) qua 4 phần:
  - **Phần A** (nghi vấn "bug độ chính xác" 6.3e-2 trước đó): xác nhận đây KHÔNG phải bug — so với
    fp32 thật, Triton (6.28e-2) chính xác HƠN cuBLAS-fp16 (1.01e-1), vì `torch.matmul(fp16,fp16)` tự
    làm tròn output về fp16 (mất chính xác), còn Triton giữ accumulator fp32 suốt. Số liệu "sai" ban
    đầu chỉ vì so sánh với 1 reference đã tự làm tròn.
  - **Phần B** (micro-scale 16×16×16): sai số = 0.0 tuyệt đối → không có bug stride/masking.
  - **Phần C — nguyên nhân thật**: dump PTX assembly cho thấy kernel Triton biên dịch ra **0 lệnh
    mma/hmma, 1024 lệnh `fma.rn.f32`** → Triton **KHÔNG hề dùng Tensor Core**, chạy hoàn toàn trên CUDA
    core thường. Đây là nguyên nhân gốc của mức chậm 178 lần.
  - **Phần D**: thử 4 cấu hình `num_warps`/`num_stages`/block size khác nhau (kể cả cấu hình mà 1 AI
    khác đề xuất) — không cấu hình nào cải thiện đáng kể, có cấu hình còn chậm hơn 232 lần → xác nhận
    đây là vấn đề compiler/codegen của Triton trên setup này, không phải vấn đề tuning.

**Kết luận Triton: dừng hẳn, có bằng chứng chắc chắn** — không phải do thiếu effort tuning.

## Custom CUDA C++ WMMA extension — vượt qua giới hạn của Triton, nhưng vẫn không đủ nhanh

Sau khi xác nhận Triton không dùng Tensor Core, viết thẳng CUDA C++ extension (`emulated_matmul_cuda/`,
build qua `torch.utils.cpp_extension`) dùng API `<mma.h>`/WMMA để tự tay kiểm soát Tensor Core, xem
Triton có phải là mắt xích hỏng hay là giới hạn phần cứng/environment chung.

**Bản v1 (naive, 1 warp/tile, đọc thẳng VRAM không qua shared memory):**
- Compile sạch, đúng cờ `-gencode=arch=compute_75,code=sm_75` (T4/Turing).
- Micro-scale (16×16×16): sai số **6.68e-06** (đạt target <1e-5, tốt hơn cả Triton).
- Full shape (8192×1536×1536): sai số **2.1e-3**, tốc độ **43.6ms/call — chậm hơn cuBLAS-fp16 25.45
  lần, chậm hơn cả fp32 thuần** (0.21x).
- **SASS check** (`cuobjdump -sass`, tìm ở `/usr/local/cuda-12.8/bin/` vì không có sẵn trên PATH):
  **80 lệnh HMMA, 0 lệnh FFMA** → xác nhận dứt khoát: WMMA viết tay DÙNG Tensor Core thật, khác hẳn
  Triton (0 HMMA). Vậy Triton hỏng do chính nó (compiler/codegen), không phải do phần cứng/environment.
- Nhưng vẫn chậm — vì kernel quá naive: đọc VRAM trực tiếp mỗi lần (latency hàng trăm chu kỳ), không
  tái sử dụng dữ liệu giữa các warp.

**Bản v2 (shared-memory tiling 64×64 tile/block + dual accumulator `acc_hi`/`acc_lo`):**
- **Bug tìm thấy khi review code do 1 AI khác đề xuất trước khi build**: bản thiết kế gốc dùng 1 cặp
  `acc_hi`/`acc_lo` DÙNG CHUNG cho cả 4 N-sub-tile trong 1 block → kết quả 4 tile khác nhau bị cộng dồn
  lẫn vào 1 accumulator, rồi ghi đè cùng 1 giá trị sai ra cả 4 vị trí output. Sửa bằng cách cấp
  `acc_hi[4]`/`acc_lo[4]` độc lập cho từng N-sub-tile.
- Sau khi sửa: micro-scale sai số **1.9e-06** (PASS), full-shape sai số **1.02e-3** (~2x tốt hơn bản
  v1, đúng như dự đoán từ kỹ thuật dual-accumulator/compensated-summation), tốc độ **22.6ms/call**
  (~1.9x nhanh hơn bản v1 nhờ shared memory), **64 HMMA / 0 FFMA** (vẫn dùng Tensor Core thật).
- **Vẫn chậm hơn cuBLAS-fp16 13.98 lần, và quan trọng hơn: vẫn chậm hơn fp32 thuần ~2.7 lần** (8.3ms
  vs 22.6ms) — nghĩa là ngay cả khi mục tiêu là "giữ độ chính xác gần fp32" (không so với fp16 nữa),
  kernel custom vẫn thua chính fp32 thuần đang có sẵn, chưa mang lại giá trị dương.
- Muốn thu hẹp khoảng cách còn lại cần kỹ thuật GEMM cấp chuyên gia hơn nữa (double buffering, tile
  lớn hơn, tránh bank conflict, vectorized load) — độ phức tạp tương đương viết lại 1 phần cuBLAS.

## Xác minh vision-only checkpoint = visual branch của full checkpoint (CONFIRMED, cả hash lẫn inference)

Mục "chưa xác nhận" ở bản trước nay đã xác nhận xong, bằng 2 lớp bằng chứng độc lập:

- **Kỹ thuật lazy-extraction** (`hash_full_clip_lazy_extract.py`, phỏng theo 1 notebook cũ của user
  từng dùng để tách text encoder — `D:\Downloads\textencoderextract.py`): xây kiến trúc `CustomTextCLIP`
  với `vision_cfg` THẬT + `text_cfg` GIẢ (near-zero, chỉ để thoả constructor), rồi dùng
  `safetensors.safe_open()` đọc lazy từng tensor `visual.*` từ file checkpoint đầy đủ — thay vì
  `open_clip.create_model_and_transforms()` (load CẢ vision+text thật + `torch.load()` toàn bộ
  checkpoint 1 lần, nghi là nguyên nhân gây crash VM, xem mục dưới).
- **Hash weight-level** (`hash_vision_only_checkpoint.py` chạy trên 1 GPU, `hash_full_clip_lazy_extract.py`
  chạy trên GPU khác — tách riêng để 2 checkpoint ~7.5GB không bao giờ cùng ở trong 1 process): sha256
  của toàn bộ 621 tensor, sort theo (shape, hash) để không phụ thuộc tên key — **khớp tuyệt đối cả 2
  lần chạy**: `a0daadff105b086bfc1c08054f9f278630e2bc33978e528c73c27ec2dda7e853`.
- **Inference-level** (cùng 5 ảnh keyframe thật, embed qua cả 2 đường): **cosine similarity = 1.0 cho
  cả 5 ảnh, max_abs_diff = 2.02e-07** (nhiễu floating-point).
- **Kết luận: vision-only checkpoint = visual branch của full checkpoint, chính xác tuyệt đối** — dùng
  thẳng vision-only checkpoint trong pipeline thật, không cần load full CLIP (tiết kiệm ~1.5GB VRAM +
  thời gian load text tower không dùng tới).

## Nguyên nhân crash VM (2 lần liên tiếp, cùng 1 script) — đã xác nhận + đã có kỹ thuật tránh

`extract_full_clip_visual_vec.py` (load full CLIP qua `open_clip.create_model_and_transforms`) làm
crash **toàn bộ VM** (không phải OOM Python bắt được — `!ps aux` cũng không chạy nổi, phải tạo session
mới) trên cả `gpu2` lẫn `gpu3` liên tiếp, dù code chỉ có 5 dòng logic (load model + embed 5 ảnh). Nghi
vấn: `create_model_and_transforms` dựng CẢ vision+text thật (random init, fp32, ~9GB) rồi
`torch.load()` toàn bộ checkpoint 1 lần vào RAM (thêm ~9GB) → đỉnh RAM hệ thống thoáng qua ~18GB, vượt
xa RAM free-tier VM (~12-13GB).

**Bằng chứng gián tiếp xác nhận giả thuyết**: kỹ thuật lazy-extraction (mục trên) — build kiến trúc với
vision thật + text GIẢ (near-zero), rồi `safetensors.safe_open()` đọc lazy từng tensor thay vì
`torch.load()` cả file — chạy xuyên suốt **866 giây tải+load với RAM hệ thống ổn định ~1.4-1.5GB free,
không crash lần nào**. Cùng file checkpoint, cùng kích thước tải, nhưng tránh được đỉnh RAM kép
(vision thật + text thật + toàn bộ checkpoint materialized) → không crash. Xác nhận nguyên nhân gốc là
cách `create_model_and_transforms` dựng model, không phải bản thân việc tải file 9GB.

**Áp dụng luôn cho Stage 1** (`bench_fp16_vs_fp32.py`): đổi loader từ `open_clip.create_model_and_transforms`
(full CLIP) sang `timm.create_model` (vision-only, đã xác nhận tương đương tuyệt đối ở mục trên) — loại
bỏ hẳn rủi ro crash thay vì chỉ hy vọng nó không lặp lại, và tận dụng luôn phát hiện tương đương ở trên.

## Stage 1 — kết quả sweep FP32 baseline / torch.compile / ONNX / FP16 (120 ảnh, đã chạy xong)

| Variant | img/s | so với FP32 | cosine_mean | cosine_min | Recall@10-overlap |
|---|---|---|---|---|---|
| FP32 baseline | 0.78 | 1x | 1.0 (ref) | 1.0 | 1.0 |
| FP32 + torch.compile(default) | 0.46 (gồm 94s warmup); ~0.72 sau warmup | 0.6x (kể cả bỏ warmup vẫn không nhanh hơn) | 1.0000001 | 1.0000001 | 1.0 |
| FP32 + ONNX export | **FAILED** (CUDA OOM, do VRAM còn giữ từ bước torch.compile trước, dọn chưa sạch) | — | — | — | — |
| **FP16** | **3.07** | **3.9x** | **0.999864** | **0.999863** | **1.0** |

**Nhận xét quan trọng**: FP16 nhanh gấp ~4 lần FP32, và **Recall@10-overlap = 1.0 tuyệt đối** trên cả
120 ảnh test — nghĩa là trong phạm vi test này, FP16 không hề làm thay đổi kết quả top-10 nearest
neighbor nào, dù cosine similarity không tuyệt đối 1.0. `torch.compile(mode="default")` KHÔNG giúp gì
(chậm hơn cả baseline, kể cả bỏ qua chi phí warmup) — log Colab báo "Not enough SMs to use
max_autotune_gemm mode", nghi T4 không đủ SM để Inductor tận dụng các đường fuse mạnh hơn.

**Ý nghĩa với nhánh CUDA kernel ở trên**: nếu mục tiêu là giữ chất lượng retrieval, FP16 trên dữ liệu
thật đã cho thấy KHÔNG mất chất lượng retrieval đo được — trong khi kernel custom (bản v2) vẫn chậm
hơn 2.7 lần so với chính FP32 thuần. Đang chờ user quyết định hướng tiếp theo dựa trên số liệu này.

## Còn dở / việc tiếp theo

- Task 5 (cross-video pipelining) và Task 6 (dọn dẹp disk qua nhiều video) — chưa bắt đầu, có thể làm
  thật với 2 session song song (vd: video A trên gpu2, video B trên gpu3).
- HF cache model PE-Core-bigG (~9GB, ở `C:\Users\duyla\.cache\huggingface\hub\...`) vẫn ở ổ C ngoài
  project — chưa chuyển vào `Data Processing`, chưa set `HF_HUB_CACHE`. Lỗi "paging file too small"
  khi chạy CPU local vẫn còn đó nếu cần dùng local (Colab thì không vấn đề gì).
- ONNX export ở Stage 1c bị OOM do dọn VRAM chưa sạch sau torch.compile — nếu muốn thử lại ONNX cần
  chạy độc lập (không chung process với bước torch.compile) hoặc gọi `gc.collect()` + `empty_cache()`
  mạnh tay hơn trước khi export.
- Kernel CUDA WMMA (bản v2) vẫn còn dư địa tối ưu (double buffering, tile lớn hơn, tránh bank conflict)
  nếu muốn tiếp tục đuổi theo — nhưng hiện tại vẫn thua fp32 thuần ~2.7 lần, chưa mang giá trị dương.
