# Track 1 — Inference Client

`track1-client` is the core client library and tooling package developed for Track 1 of the AI Inference Engineering curriculum.

---

## Features

- **Typed Data Structures**: Strongly typed Python dataclasses for inference configuration:
  - `InferenceRequest`: Complete request payload including prompt, model, generation settings, and server endpoints.
  - `GenerationConfig`: Hyperparameters such as `max_tokens`, `temperature`, `top_p`, and stop sequences.
  - `ServerConfig`: Endpoint configuration including host, port, protocol (`http`, `https`, `grpc`), and timeouts.
- **Strict Validation**: Explicit validation rules with domain-specific error reporting (`ConfigValidationError`).
  - Guards against invalid or non-positive token bounds (`0 < max_tokens <= 32768`).
  - Validates temperature `[0.0, 2.0]` and top_p `(0.0, 1.0]`.
  - Validates network port ranges (`1 - 65535`) and valid protocols.
  - Enforces mandatory fields and rejects blank strings.
- **CLI Utility (`infer-config`)**: Inspect, validate, and emit inference configurations directly from your terminal.
- **Robust Testing**: Comprehensive test suite covering valid payloads, edge cases, malformed JSON, and CLI flags.

---

## Installation

Ensure you are inside the virtual environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Unix / macOS
source .venv/bin/activate
```

Install the package in editable mode:

```powershell
pip install -e .
```

---

## Python Usage Example

```python
from inference_client import (
    ConfigValidationError,
    GenerationConfig,
    InferenceRequest,
    ServerConfig,
    parse_config_json,
)

# 1. Direct programmatic instantiation
request = InferenceRequest(
    prompt="Explain continuous batching in vLLM.",
    model="meta-llama/Llama-3-8B-Instruct",
    generation=GenerationConfig(max_tokens=256, temperature=0.7),
    server=ServerConfig(host="127.0.0.1", port=8000),
)
print(request.to_json())

# 2. Parsing from JSON string or file
json_payload = '{"prompt": "Hello", "model": "my-model", "generation": {"max_tokens": 128}}'
parsed_request = parse_config_json(json_payload)
print(f"Loaded request for model: {parsed_request.model}")
```

---

## CLI Usage (`infer-config`)

The package registers the `infer-config` command console script:

### Generate sample configuration:
```powershell
infer-config --sample
```

### Validate a configuration file:
```powershell
infer-config config.json
```

### Silent validation check (for scripts / CI):
```powershell
infer-config --config config.json --validate-only
echo $LASTEXITCODE
```

### Pipe JSON via stdin:
```powershell
Get-Content config.json | infer-config
```

---

## Running Tests

Run the test suite using Python's built-in `unittest` runner:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Or with `pytest` (if installed):

```powershell
pytest
```
