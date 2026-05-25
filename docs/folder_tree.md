aic2026-vbs/
│
├── remote-server/                  # 1. TRIỂN KHAI TRÊN MÁY TRẠM GPU (PRODUCTION)
│   ├── docker-compose.yml          # Quản lý Docker cho Milvus + PostgreSQL
│   ├── requirements.txt
│   ├── main.py                     # FastAPI Server chính (Chạy với ENV_MODE=SERVER)
│   └── app/
│       ├── db/                     # Kết nối cơ sở dữ liệu gốc (Native SDKs)
│       │   ├── milvus_client.py
│       │   └── postgres_client.py
│       │
│       ├── data_provider.py        # Chế độ SERVER: Đọc/Ghi trực tiếp từ DB local ra RAM
│       │
│       └── strategies/             # NƠI CHỨA CÁC CHIẾN THUẬT ĐÃ NGHIỆM THU (BẢN STABLE)
│           ├── __init__.py
│           ├── base_strategy.py    # Class trừu tượng mẫu (chứa Timeout + Fetch Cap)
│           └── stable_fusion.py    # <--- File chiến thuật chuẩn (Bê nguyên xi từ local lên)
│
└── local-client/                   # 2. CHẠY TRÊN LAPTOP CỦA THÀNH VIÊN TRONG ĐỘI (LOCAL)
    │
    ├── frontend/                   # Web UI (Next.js / React - Chỉ làm giao diện)
    │   ├── package.json
    │   └── src/
    │       ├── components/         # VideoPlayer (YouTube API), SearchSemantic, SearchText...
    │       └── pages/index.tsx     # Giao diện chính, gọi API về localhost:8000
    │
    └── local-backend/              # SÂN CHƠI PHÁT TRIỂN THUẬT TOÁN PYTHON (LOCAL BACKEND)
        ├── requirements.txt
        ├── main.py                 # FastAPI Local (Chạy với ENV_MODE=LOCAL)
        └── app/
            ├── data_provider.py    # Chế độ LOCAL: Đóng vai trò Proxy gọi API qua Ngrok/LAN sang Server để lấy dữ liệu thô
            │
            └── strategies/         # NƠI ANH EM TỰ DO CODE & THỬ NGHIỆM THUẬT TOÁN MỚI
                ├── __init__.py
                ├── base_strategy.py # Giống hệt file base trên Server để đảm bảo tính đồng bộ
                ├── stable_fusion.py # Bản stable hiện tại để đối chiếu điểm số
                ├── duy_temporal_v1.py # Duy tự tạo file này để test thuật toán của mình
                └── nam_matrix_v2.py   # Nam tự tạo file này để test thuật toán của mình