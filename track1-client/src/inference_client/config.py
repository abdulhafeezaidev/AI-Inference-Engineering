"""Structured configuration and validation for AI inference requests."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class ConfigValidationError(ValueError):
    """Raised when an inference configuration fails validation."""

    def __init__(self, message: str, field_name: Optional[str] = None) -> None:
        super().__init__(message)
        self.field_name = field_name
        self.message = message

    def __str__(self) -> str:
        if self.field_name:
            return f"Validation error for '{self.field_name}': {self.message}"
        return f"Validation error: {self.message}"


class ProtocolType(str, Enum):
    """Supported communication protocols."""

    HTTP = "http"
    HTTPS = "https"
    GRPC = "grpc"


@dataclass
class ServerConfig:
    """Connection settings for the model inference server."""

    host: str = "127.0.0.1"
    port: int = 8000
    protocol: ProtocolType = ProtocolType.HTTP
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.host or not isinstance(self.host, str):
            raise ConfigValidationError("Server host must be a non-empty string", "host")

        if not isinstance(self.port, int) or not (1 <= self.port <= 65535):
            raise ConfigValidationError(
                f"Server port must be an integer between 1 and 65535 (got {self.port})",
                "port",
            )

        if isinstance(self.protocol, str):
            try:
                self.protocol = ProtocolType(self.protocol.lower())
            except ValueError:
                valid_protocols = [p.value for p in ProtocolType]
                raise ConfigValidationError(
                    f"Invalid protocol '{self.protocol}'. Expected one of: {valid_protocols}",
                    "protocol",
                )

        if not isinstance(self.timeout_seconds, (int, float)) or self.timeout_seconds <= 0:
            raise ConfigValidationError(
                f"timeout_seconds must be a positive number (got {self.timeout_seconds})",
                "timeout_seconds",
            )

    @property
    def base_url(self) -> str:
        return f"{self.protocol.value}://{self.host}:{self.port}"


@dataclass
class GenerationConfig:
    """Hyperparameters and generation constraints for inference."""

    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool):
            raise ConfigValidationError(
                f"max_tokens must be an integer (got {type(self.max_tokens).__name__})",
                "max_tokens",
            )

        if self.max_tokens <= 0:
            raise ConfigValidationError(
                f"max_tokens must be greater than 0 (got {self.max_tokens})",
                "max_tokens",
            )

        if self.max_tokens > 32768:
            raise ConfigValidationError(
                f"max_tokens exceeds maximum allowed limit of 32768 (got {self.max_tokens})",
                "max_tokens",
            )

        if not isinstance(self.temperature, (int, float)) or self.temperature < 0.0 or self.temperature > 2.0:
            raise ConfigValidationError(
                f"temperature must be between 0.0 and 2.0 (got {self.temperature})",
                "temperature",
            )

        if not isinstance(self.top_p, (int, float)) or not (0.0 < self.top_p <= 1.0):
            raise ConfigValidationError(
                f"top_p must be in the range (0.0, 1.0] (got {self.top_p})",
                "top_p",
            )

        if not isinstance(self.stop_sequences, list):
            raise ConfigValidationError(
                "stop_sequences must be a list of strings",
                "stop_sequences",
            )


@dataclass
class InferenceRequest:
    """Complete structured specification for an inference call."""

    prompt: str
    model: str
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    request_id: Optional[str] = None
    stream: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ConfigValidationError(
                "prompt must be a non-empty string",
                "prompt",
            )

        if not isinstance(self.model, str) or not self.model.strip():
            raise ConfigValidationError(
                "model must be a non-empty string identifier",
                "model",
            )

        if not isinstance(self.generation, GenerationConfig):
            raise ConfigValidationError(
                "generation must be an instance of GenerationConfig",
                "generation",
            )
        self.generation.validate()

        if not isinstance(self.server, ServerConfig):
            raise ConfigValidationError(
                "server must be an instance of ServerConfig",
                "server",
            )
        self.server.validate()

    def to_dict(self) -> dict[str, Any]:
        """Convert request into a JSON-serializable dictionary."""
        data = asdict(self)
        data["server"]["protocol"] = self.server.protocol.value
        return data

    def to_json(self, indent: int = 2) -> str:
        """Convert request to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def parse_config_dict(data: dict[str, Any]) -> InferenceRequest:
    """Parse and validate an inference configuration from a Python dictionary."""
    if not isinstance(data, dict):
        raise ConfigValidationError("Configuration root must be a JSON object / dictionary")

    # Required top-level keys check
    if "prompt" not in data:
        raise ConfigValidationError("Missing required field 'prompt'", "prompt")
    if "model" not in data:
        raise ConfigValidationError("Missing required field 'model'", "model")

    # Server config parsing
    server_data = data.get("server", {})
    if not isinstance(server_data, dict):
        raise ConfigValidationError("'server' field must be an object", "server")
    server_config = ServerConfig(**server_data)

    # Generation config parsing
    gen_data = data.get("generation", {})
    if not isinstance(gen_data, dict):
        raise ConfigValidationError("'generation' field must be an object", "generation")
    generation_config = GenerationConfig(**gen_data)

    return InferenceRequest(
        prompt=data["prompt"],
        model=data["model"],
        generation=generation_config,
        server=server_config,
        request_id=data.get("request_id"),
        stream=bool(data.get("stream", False)),
    )


def parse_config_json(json_str: str) -> InferenceRequest:
    """Parse and validate an inference configuration from a JSON string."""
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as err:
        raise ConfigValidationError(f"Invalid JSON syntax: {err}") from err

    return parse_config_dict(data)


def load_config_from_file(file_path: str | Path) -> InferenceRequest:
    """Load, parse, and validate an inference configuration from a JSON file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    content = path.read_text(encoding="utf-8")
    return parse_config_json(content)
