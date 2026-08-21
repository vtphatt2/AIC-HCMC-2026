# AIC26 — Thể lệ nộp bài sơ tuyển

Nguồn: https://sotuyenaic.oj.io.vn/rules/ (lưu lại 2026-08-21 vì trang chặn
fetch tự động — không có API/RSS để đồng bộ, copy thủ công khi thể lệ đổi).

## Các loại truy vấn

Vòng sơ tuyển gồm 3 dạng truy vấn:

- **Textual Known Item Search (KIS)** — tìm kiếm chính xác theo văn bản
- **Visual Question Answering (Q&A)** — truy vấn dạng Hỏi-Đáp
- **Temporal Retrieval and Alignment of Key Events (TRAKE)** — truy xuất và
  căn chỉnh sự kiện video theo thời gian

## Gói truy vấn và tên file — điểm mấu chốt

BTC cấp từng gói truy vấn theo đợt, mỗi câu truy vấn có 1 file text riêng,
ví dụ đợt 1 gồm 4 câu: `query-1-kis.txt`, `query-2-kis.txt`,
`query-3-qa.txt`, `query-4-trake.txt`.

**Tên file CSV nộp phải khớp chính xác với tên truy vấn BTC cấp** —
`query-{số}-{loại}.csv`, số và loại không tự chọn mà theo đúng file BTC
đã đưa cho câu đó. Hậu tố: `kis` / `qa` / `trake`.

**Mỗi câu truy vấn nộp đúng 1 file CSV** — không thể có 2 file cùng
`query-{số}-{loại}.csv` trong 1 lần nộp.

## Format từng dòng CSV

Không có header row, delimiter dấu phẩy, encoding UTF-8, line ending CRLF
hoặc LF. Tối đa 100 dòng/file. Tên video **không** có đuôi `.mp4`.

**KIS**: `<video_name>, <frame_id>`
```
L00_V000, 1234
L00_V055, 5555
```

**Q&A**: `<video_name>, <frame_id>, <answer>` — answer tối đa 100 ký tự, VI
hoặc EN, so sánh chính xác ngữ nghĩa với đáp án. Ngoặc kép **bắt buộc**
khi answer chứa dấu phẩy, ngoặc kép, hoặc xuống dòng (ngoặc kép bên trong
escape bằng `""`); answer đơn giản thì không bắt buộc, nhưng luôn đặt
ngoặc kép cho mọi answer cũng được chấp nhận.
```
L01_V028, 3450, "5"
L02_V011, 1200, "Năm người"
L03_V005, 2800, "Màu đỏ, rất đẹp"
```

**TRAKE**: `<video_name>, <frame_1>, <frame_2>, ..., <frame_N>` — số
lượng Frame ID phải khớp đúng số events yêu cầu, thứ tự theo thời gian.
```
L10_V001, 1200, 1850, 2100, 2450
```

## Đóng gói nộp bài

1. Tạo thư mục tên đúng `submission`
2. Đặt tất cả CSV kết quả vào trong đó
3. Nén **thư mục `submission`** thành `.zip` — không nén trực tiếp các
   file CSV (thiếu thư mục bọc ngoài là lỗi thường gặp)
4. (Tùy chọn) đổi tên file zip tùy ý, khuyến cáo chỉ dùng chữ/số

```
submission/
├── query-1-kis.csv
├── query-2-kis.csv
├── query-3-qa.csv
└── query-4-trake.csv
```

Chỉ chấp nhận `.csv` thật (không phải Excel `.xlsx` đổi đuôi) — mở bằng
Notepad phải thấy text thuần, không ký tự lạ.

## Nộp bài & chấm điểm

- Đăng nhập tài khoản BTC đã cấp, nộp `.zip` trực tiếp trên hệ thống thi
- Public Leaderboard: 50% đáp án BTC. Private/xếp hạng cuối: 100% đáp án
- **Tối đa 3 lần nộp** mỗi gói truy vấn — tính theo **lần nộp cuối cùng**
- Nộp sai định dạng vẫn tính là 1 lần nộp
- Mỗi đội chỉ dùng 1 tài khoản để nộp

## Liên hệ codebase

`local-client/frontend/src/lib/submission/index.ts` (`buildSubmissionCsv`,
`buildSubmissionZip`) implement đúng format ở trên. Tên session **là** tên
file export (`{session}.csv`) — trùng tên bị chặn ngay lúc tạo session
(409), nên việc 2 file trùng tên trong 1 lần zip là bất khả thi về cấu
trúc, không cần check riêng lúc build zip nữa. Xem thêm
[SUBMISSION.md](SUBMISSION.md) cho cách dùng dashboard.
