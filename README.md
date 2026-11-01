# AI Inference Engineering & Client Infrastructure

A production-grade repository for AI inference systems, low-latency LLM serving client architecture, and streaming telemetry benchmarking.

---

## Architecture Overview

```text
+-----------------------------------------------------------------------------------+
|                        Inference Client Infrastructure (Python)                   |
|                                                                                   |
|  [ Typed Config Layer ]             [ Asynchronous Streaming Engine ]             |
|  - Pydantic / Dataclass validation  - Async HTTPX client connection pooling       |
|  - Timeout & retry policies         - Server-Sent Events (SSE) token stream parser|
|                                                                                   |
|  [ Real-Time Telemetry & Metrics ]  [ Resilient Error Classifier ]                |
|  - Time-To-First-Token (TTFT) clock - HTTPStatusError & ConnectError recovery     |
|  - End-to-End latency breakdown     - Graceful chunk cancellation                 |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                   Model Serving & Deployment Runtime (Docker / GPU)               |
|                                                                                   |
|  - NVIDIA Container Toolkit & CUDA integration (`docker-compose.gpu.yml`)         |
|  - High-throughput token generation benchmarks and validation gates               |
|  - Standardized request/response inference payload protocols                      |
+-----------------------------------------------------------------------------------+
```

---

## Key Capabilities

1. **Low-Latency Streaming Client (`track1-client`)**:
   - Asynchronous streaming token ingestion powered by `httpx` and `asyncio`.
   - Real-time chunk timing capturing nanosecond-accurate Time-To-First-Token (TTFT) and end-to-end token latency.
   - Comprehensive error handling matrix covering socket timeouts, HTTP status errors, and stream aborts.

2. **Benchmarking & Validation Artifacts**:
   - Structured experimental runs and trace files in `artifacts/` tracking inference efficiency.
   - Self-contained GPU deployment recipes (`docker-compose.gpu.yml`).

3. **Inference Engineering Systems Curriculum**:
   - Comprehensive architecture breakdowns, token economics, and low-level attention mechanism implementations (`day17_attention.py`).

---

## Technology Stack

- **Language**: Python 3.12+ / 3.14
- **Async Runtime**: Python `asyncio`, `httpx`
- **Data Modeling & Validation**: Python `dataclasses`, `pydantic`
- **Containerization & Compute**: Docker, Docker Compose, NVIDIA CUDA GPU acceleration
- **Testing**: `pytest`, `pytest-asyncio`

---

## Project Structure

```text
AI-Inference-Engineering/
├── track1-client/                       # Core inference client package
│   ├── src/
│   │   └── inference_client/
│   │       ├── client.py                # Async streaming client & TTFT timer
│   │       ├── config.py                # Typed client configuration
│   │       ├── cli.py                   # Command-line benchmarking tool
│   │       └── async_demo.py            # Concurrent benchmark runner
│   ├── tests/                           # Unit & async integration tests
│   ├── Dockerfile                       # Containerized client runtime
│   ├── docker-compose.gpu.yml           # GPU-accelerated server environment
│   └── pyproject.toml
├── artifacts/                           # Daily telemetry, benchmarks & logs
├── Resources/                           # Reference guides & architectural specs
├── CodeWork/                            # Prototyping scripts
└── README.md
```

---

## Getting Started

### Installation
```bash
cd track1-client
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Running Streaming Benchmarks
```powershell
$env:PYTHONPATH = "src"
python -m inference_client.cli --url http://127.0.0.1:8765/stream --prompt "Explain transformer attention mechanics."
```

### Running Test Suite
```powershell
$env:PYTHONPATH = ".;src"
pytest tests -v
```
