# AI Inference Engineering

A structured repository for practical AI Inference Systems, ML Infrastructure, and Model Serving engineering.

---

## Architecture Overview

```text
AI Inference Engineering/
│
├── .gitignore
├── README.md
│
├── artifacts/
│   ├── day01/
│   └── day02/
│
├── Resources/
├── CodeWork/
│
└── track1-client/
    ├── .venv/
    ├── pyproject.toml
    │
    ├── src/
    │   └── inference_client/
    │       ├── __init__.py
    │       ├── config.py
    │       └── cli.py
    │
    ├── tests/
    │   └── test_config.py
    │
    └── README.md
```

---

## Directory Layout

- **`artifacts/`**: Dated daily deliverables, benchmark measurements, execution logs, and experiment traces (e.g., `day01/`, `day02/`).
- **`Resources/`**: Curriculum roadmaps, reference guides, book compilation scripts, and specifications.
- **`CodeWork/`**: Prototyping scripts and configuration exploration snippets.
- **`track1-client/`**: The primary inference client package implementing typed configuration, asynchronous streaming, connection management, metrics collection, and CLI tools.

---

## Getting Started

### 1. Prerequisites
- Python >= 3.12 (Python 3.14 supported)
- Windows PowerShell or Unix-like bash shell

### 2. Setting Up `track1-client`
Navigate to the `track1-client/` directory and activate the virtual environment:

```powershell
cd track1-client

# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# On Unix / macOS:
source .venv/bin/activate
```

Install the package in editable mode:
```powershell
python -m pip install -e .
```

### 3. Running CLI Tools
Test configuration parsing and validation using the `infer-config` command:
```powershell
infer-config --sample
```

### 4. Running Tests
Run the test suite:
```powershell
python -m unittest discover -s tests
```
