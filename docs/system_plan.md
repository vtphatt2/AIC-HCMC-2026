Act as a Senior Software Architect and Lead AI System Engineer. I need you to bootstrap a complete project blueprint and foundational boilerplate code for a video retrieval system designed for the Ho Chi Minh City AI Challenge (similar to Video Browser Showdown / Lifelog Challenge).

This system must support a dual-mode execution environment: a local "Playground" for team members to develop and test Python algorithms independently, and a seamless deployment pipeline to promote those exact same Python strategy files to the production GPU Server without changing any algorithmic code.

---

## 1. DUAL-MODE SYSTEM ARCHITECTURE
We need two main deployment components, both running FastAPI in Python to ensure code compatibility:

1. **Remote Production Server (GPU Workstation):**
   - Hosts Milvus (CLIP embeddings) and PostgreSQL (OCR & Transcripts).
   - Can run in two modes based on an environment variable (`ENV_MODE=SERVER` or `ENV_MODE=LOCAL_PLAYGROUND`).
   - In Server Mode, it directly orchestrates the final production strategies and serves the Frontend.

2. **Local Client App (Contestant's Laptop):**
   - **Frontend (TypeScript / Next.js):** Handles UI rendering, Search Box inputs (Semantic box with Google Translate toggle, and Text box for OCR/Transcript), and high-precision YouTube IFrame Player navigation (`Seconds = Frame_ID / Video_FPS`).
   - **Local Backend (Python / FastAPI):** The development playground. It runs exactly the same strategy architecture as the server.

---

## 2. THE DESIGN PATTERN: DATA ACCESS ABSTRACTION
To achieve "Write Once, Run Anywhere" for our strategy files, you must implement a `DataProvider` abstraction layer:
- When a strategy requests raw data (visual vectors, OCR text, or transcript intervals), it calls `DataProvider.get_raw_data()`.
- **If `ENV_MODE == "LOCAL"` (On Laptop):** The `DataProvider` internally acts as an HTTP/gRPC client, fetching raw data from the Remote Server via an Ngrok/LAN URL.
- **If `ENV_MODE == "SERVER"` (On GPU Workstation):** The `DataProvider` connects directly to the local Milvus and PostgreSQL databases using native SDKs for maximum speed and zero network latency.

This ensures that the core algorithmic method `fusion_and_temporal(raw_data)` remains 100% untouched when copying a tested script from the client's laptop straight into the server's production environment.

---

## 3. DATA MODALITIES & PLUGGABLE STRATEGY GUARDRAILS
The strategies merge three types of data on RAM: Frame-level visual vectors, Frame-level OCR tokens, and Interval-level Transcripts (`start_time_ms` to `end_time_ms`).

### System Guardrails (Applied globally via the Base Strategy Class):
- **Fetch Cap:** Enforce a strict maximum limit of 1,000 raw records fetched per database query to prevent memory overflow during local prototyping.
- **Execution Timeout:** Wrap the algorithm execution step in a strict 2.0-second timeout wrapper. If a team member introduces an infinite loop in their experimental script, the request safely terminates without crashing the engine.

---

## YOUR TASK:
Please write clean, well-commented, production-ready Python and TypeScript boilerplate matching this architecture:

1. **Project Directory Layout:** Show an organized tree layout demonstrating where shared `strategies/` are located on both client and server sides.
2. **The Abstract Data Provider (`data_provider.py`):** Implement the conditional switching logic based on `ENV_MODE` (fetching via API requests vs fetching via direct DB connections).
3. **The Base Abstract Strategy Class (`base_strategy.py`):** Define the plugin lifecycle methods (`pre_process`, `execute`, `fusion_and_temporal`, `post_filter`) equipped with the 2-second execution timeout guardrail.
4. **An Experimental Strategy Example (`team_strategy_v1.py`):** Provide a sample implementation showing how to combine interval-based transcripts with frame-level visual data on RAM using basic Python data structures.
5. **The Dynamic FastAPI Strategy Router (`main.py`):** Write the endpoint logic that automatically detects and instantiates the chosen strategy based on a `strategy_id` passed from the frontend selection dropdown.
6. **PostgreSQL DDL Schemas:** Provide schemas for Video Metadata, OCR, and Transcripts with composite indexes optimized for fast range and interval queries.