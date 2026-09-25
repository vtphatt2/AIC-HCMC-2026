# Chạy VORTA với Search Agent và Verify Agent

Hướng dẫn này dùng code và môi trường đã có trong checkout. Không cần cài thêm package để bật agent. Agent là dịch vụ tùy chọn: tìm kiếm thủ công vẫn gọi VORTA backend trực tiếp nếu agent dừng hoặc lỗi.

## 1. Nối ba tiến trình

| Tiến trình | Biến/cổng mặc định | Vai trò |
|---|---|---|
| VORTA backend | `127.0.0.1:8000` | Search, transcript, frame/context và media hiện có |
| Agent service | `VORTA_BACKEND_URL=http://127.0.0.1:8000`, cổng `8012` | Một Search Agent và một hàng đợi Verify Agent trong RAM; gọi Codex CLI |
| Next.js frontend | `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000`, `AGENT_SERVER_URL=http://127.0.0.1:8012`, cổng `3000` | Manual search gọi backend; `/api/agent-search` và `/api/agent-verify` chuyển tiếp đến agent service |

`NEXT_PUBLIC_API_URL` phải truy cập được từ **browser** của operator. `AGENT_SERVER_URL` chỉ cần truy cập được từ **máy chạy Next.js**; browser không gọi cổng 8012. `VORTA_BACKEND_URL` phải truy cập được từ **máy chạy agent service**. Nếu ba tiến trình ở các máy khác nhau, đặt từng URL theo vị trí tương ứng.

Không có bước đăng ký MCP. Agent service dùng Codex CLI đã đăng nhập để lập kế hoạch/đánh giá, rồi gọi HTTP API của VORTA hiện có. Hai role dùng hai context Codex tách biệt.

Agent service dùng state trong RAM: chỉ chạy **một process uvicorn**, không thêm `--workers`. Khởi động lại agent service sẽ xóa query, queue và kết quả agent; manual search không bị xóa.

## 2. Kiểm tra môi trường sẵn có

Chạy từ thư mục gốc repository:

```bash
test -x local-client/local-backend/.venv/bin/python
test -d local-client/frontend/node_modules
codex --version
codex login status
```

Backend và agent đều dùng `local-client/local-backend/.venv/bin/python`; không cần kích hoạt venv nếu gọi đúng đường dẫn. `codex` phải có trên `PATH` và đã đăng nhập dưới cùng tài khoản hệ điều hành chạy agent service. Các lệnh trên chỉ kiểm tra, không cài gì.

## 3. Chạy nhanh trên một máy (Mac/Linux/WSL)

```bash
bash scripts/start-local.sh
```

Script khởi động backend `8000`, agent service `8012` và frontend `3000`. Mở `http://127.0.0.1:3000`. Có thể đổi cổng bằng `--backend-port`, `--agent-port`, `--frontend-port`; chọn encoder bằng `--backend` như [SETUP.md](../SETUP.md#4-one-command-launch-scripts). Script bỏ qua dịch vụ đã nghe trên cổng tương ứng, nên nếu frontend cũ chạy với URL khác, hãy khởi động lại frontend để nhận biến môi trường mới.

Nếu không có terminal emulator, script ghi log vào `challenge_resources/runtime-logs/{backend,search-agent,frontend}.log` và `.err.log` tương ứng.

## 4. Gắn agent vào VORTA đang chạy

Giữ VORTA backend hiện tại. Trong terminal mới, từ thư mục gốc repository:

```bash
VORTA_BACKEND_URL=http://127.0.0.1:8000 \
  local-client/local-backend/.venv/bin/python -m uvicorn agent.server:app \
  --host 127.0.0.1 --port 8012
```

Nếu backend ở máy khác, thay `VORTA_BACKEND_URL` bằng URL backend mà máy agent truy cập được. Sau đó khởi động **hoặc khởi động lại** frontend:

```bash
cd local-client/frontend
AGENT_SERVER_URL=http://127.0.0.1:8012 \
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 \
npm run dev -- -H 127.0.0.1 -p 3000
```

Nếu frontend và agent ở hai máy khác nhau, dùng địa chỉ agent truy cập được từ máy frontend cho `AGENT_SERVER_URL` và bind agent vào giao diện mạng phù hợp. Giữ cổng agent trong mạng tin cậy; operator chỉ cần URL frontend.

Trên Windows PowerShell, `scripts/start-local.ps1` hiện chỉ khởi động backend/frontend. Đặt `AGENT_SERVER_URL` trước khi chạy script, rồi khởi động agent riêng từ repo root bằng venv Windows:

```powershell
$env:AGENT_SERVER_URL = "http://127.0.0.1:8012"
scripts\start-local.ps1

# Terminal PowerShell khác, từ repo root:
$env:VORTA_BACKEND_URL = "http://127.0.0.1:8000"
& .\local-client\local-backend\.venv\Scripts\python.exe -m uvicorn agent.server:app --host 127.0.0.1 --port 8012
```

## 5. Hai operator qua LAN hoặc tunnel

Trên máy chạy cả ba tiến trình, cho hai browser truy cập cùng một frontend:

```bash
bash scripts/start-local.sh --lan-address 192.168.0.102
```

Operator mở `http://192.168.0.102:3000`. Manual search là state riêng trong từng browser; Search Agent và Verify Agent là state chung trên agent service. Agent vẫn bind `127.0.0.1:8012` vì Next.js ở cùng máy sẽ proxy yêu cầu của cả hai browser.

Với ngrok/Cloudflare, dùng `NEXT_PUBLIC_API_URL=/`, để `AGENT_SERVER_URL` trỏ đến agent service từ máy Next.js, rồi đưa tunnel tới `scripts/share-proxy.cjs` ở cổng `3001`. Proxy chuyển `/api/agent-search` và `/api/agent-verify` tới Next.js; search/media `/api/*` còn lại tới VORTA backend. Các lệnh tunnel và cấu hình CORS có trong [SETUP.md](../SETUP.md#5-sharing-across-machines-ngrok-or-cloudflare-tunnel). Khởi động lại frontend sau khi đổi `NEXT_PUBLIC_API_URL`.

## 6. Dùng trong UI

1. Operator có thể tìm kiếm thủ công như trước, kể cả khi agent đang chạy.
2. Dán official query vào ô **Search Agent** (hoặc chọn **Use manual text**), rồi bấm **Agent Search**. Kết quả xuất hiện trong panel agent; **Cancel** hủy job, **Reset** xóa query và cả trạng thái Verify liên quan.
3. Trên một kết quả thủ công hoặc agent, bấm **Verify with Agent**. Verify chỉ xử lý candidate được người bấm chọn; hai operator dùng chung một queue. **Cancel Verify** và **Reset Verify** chỉ tác động đến Verify.
4. Đọc từng điều kiện `MATCH`, `MISMATCH`, `UNKNOWN`, mở video để kiểm tra và tự quyết định. Hiện Verify chỉ dùng transcript/metadata; mọi điều kiện cần nhìn ảnh luôn là `UNKNOWN` cho tới khi test truyền ảnh thật hoàn tất. Không có tự động gửi DRES.

Khi bắt đầu official query mới bằng Agent Search, hệ thống hủy/bỏ qua job cũ, xóa kết quả agent và queue Verify. Yêu cầu Verify mang query cũ bị từ chối bằng HTTP `409`. Đừng chạy nhiều agent service process vì chúng không chia sẻ RAM.

## 7. Kiểm tra kết nối và lỗi thường gặp

```bash
curl -fsS http://127.0.0.1:8000/api/health
curl -fsS http://127.0.0.1:8012/api/agent/search
curl -fsS http://127.0.0.1:3000/api/agent-search
curl -fsS http://127.0.0.1:3000/api/agent-verify
```

Các endpoint agent GET trả JSON có `status: "idle"` khi chưa chạy query. Muốn kiểm tra Search Agent ngoài UI với dataset đang sẵn sàng:

```bash
local-client/local-backend/.venv/bin/python -m agent.app \
  "a person riding a bicycle" --base-url http://127.0.0.1:8000 --top-k 3
```

| Hiện tượng | Kiểm tra |
|---|---|
| Panel agent báo Unavailable / HTTP 503 | Agent service cổng `8012` đang chạy? `AGENT_SERVER_URL` đúng từ máy Next.js? Xem log `search-agent` nếu script chạy nền. Manual search vẫn dùng được. |
| Agent báo không gọi được Codex | Chạy `codex --version` và `codex login status` dưới cùng user chạy agent service. |
| Agent báo lỗi VORTA hoặc chỉ còn `UNKNOWN` | Kiểm tra `VORTA_BACKEND_URL`, `/api/health`, transcript và frame/context endpoint. Thiếu bằng chứng không được xem là `MISMATCH`. |
| Verify báo query cũ / HTTP 409 | Dùng official query hiện tại trong Search Agent hoặc Reset để bắt đầu query mới. |
| Tunnel mở trang nhưng không thấy agent | Kiểm tra tunnel tới proxy `3001`, proxy đã chạy phiên bản mới, frontend đã được khởi động lại với `NEXT_PUBLIC_API_URL=/`. |

Trạng thái triển khai/test hiện tại: [PROGRESS.md](PROGRESS.md). Thiết kế và giới hạn MVP: [MVP.md](MVP.md).
