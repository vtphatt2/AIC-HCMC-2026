# Repository Folder Structure

```text
aic2026-vbs/
│
├── remote-server/                  # Production GPU server deployment
│   ├── docker-compose.yml          # Docker composition for Milvus + PostgreSQL
│   ├── requirements.txt
│   ├── main.py                     # Main FastAPI Server entry point (runs with ENV_MODE=SERVER)
│   └── app/
│       ├── db/                     # Native database connection layers
│       │   ├── milvus_client.py
│       │   └── postgres_client.py
│       │
│       ├── data_provider.py        # Direct database query orchestration layer
│       │
│       └── strategies/             # Verified production-ready search strategies
│           ├── __init__.py
│           ├── base_strategy.py    # Abstract base strategy (enforces timeouts & fetch caps)
│           └── stable_fusion.py    # Stable multi-modal fusion strategy baseline
│
└── local-client/                   # Local developer workspace components
    │
    ├── frontend/                   # React / Next.js client application
    │   ├── package.json
    │   └── src/
    │       ├── components/         # Interactive UI components (VideoPlayer, Search inputs)
    │       └── pages/index.tsx     # Main dashboard interface
    │
    └── local-backend/              # Local Python playground backend (runs with ENV_MODE=LOCAL/MOCK)
        ├── requirements.txt
        ├── main.py                 # Local FastAPI server entry point
        └── app/
            ├── data_provider.py    # Data proxy layer (routes calls to Remote Server or local mock JSON)
            │
            └── strategies/         # Strategy development and prototyping directory
                ├── __init__.py
                ├── base_strategy.py # Base class (identical to server base class for compatibility)
                ├── stable_fusion.py # Copy of production stable strategy for benchmarking
                └── custom_strategy_v1.py # Custom developer strategy implementation
```