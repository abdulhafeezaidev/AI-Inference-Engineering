"""Command-line interface for inference configuration inspection and validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import (
    ConfigValidationError,
    GenerationConfig,
    InferenceRequest,
    ProtocolType,
    ServerConfig,
    load_config_from_file,
    parse_config_json,
)


def create_sample_request() -> InferenceRequest:
    """Generate a valid sample inference request."""
    return InferenceRequest(
        prompt="Explain KV-cache memory calculation in modern LLM serving.",
        model="meta-llama/Llama-3-8B-Instruct",
        generation=GenerationConfig(
            max_tokens=512,
            temperature=0.7,
            top_p=0.95,
            stop_sequences=["<|eot_id|>"],
        ),
        server=ServerConfig(
            host="127.0.0.1",
            port=8000,
            protocol=ProtocolType.HTTP,
            timeout_seconds=60.0,
        ),
        request_id="req-sample-001",
        stream=True,
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for infer-config."""
    parser = argparse.ArgumentParser(
        prog="infer-config",
        description="Validate, inspect, and generate AI inference configurations.",
    )
    parser.add_argument(
        "config_file",
        nargs="?",
        default=None,
        help="Path to JSON configuration file (or '-' to read from stdin).",
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config_opt",
        help="Path to JSON configuration file.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Print a default valid sample configuration JSON.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate configuration silently, exiting 0 if valid or 1 if invalid.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for infer-config CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.sample:
        sample = create_sample_request()
        print(sample.to_json(indent=2))
        return 0

    target_path = args.config_opt or args.config_file

    if not target_path:
        if not sys.stdin.isatty():
            target_path = "-"
        else:
            parser.print_help(sys.stderr)
            return 1

    try:
        if target_path == "-":
            raw_data = sys.stdin.read()
            request = parse_config_json(raw_data)
        else:
            request = load_config_from_file(Path(target_path))

        if args.validate_only:
            print("OK: Configuration is valid.", file=sys.stdout)
        else:
            print(f"Validation successful for model '{request.model}'")
            print(request.to_json(indent=2))

        return 0

    except FileNotFoundError as err:
        print(f"File Error: {err}", file=sys.stderr)
        return 1
    except ConfigValidationError as err:
        print(f"Configuration Error: {err}", file=sys.stderr)
        return 1
    except Exception as err:
        print(f"Unexpected Error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
