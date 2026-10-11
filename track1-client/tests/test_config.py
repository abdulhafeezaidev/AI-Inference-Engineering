"""Unit tests for inference client configuration parsing and validation."""

import io
import json
import tempfile
import unittest
from pathlib import Path

from inference_client.cli import main
from inference_client.config import (
    ConfigValidationError,
    GenerationConfig,
    InferenceRequest,
    ProtocolType,
    ServerConfig,
    load_config_from_file,
    parse_config_dict,
    parse_config_json,
)


class TestConfigValidation(unittest.TestCase):
    """Test suite for config dataclasses and parsing logic."""

    def test_default_configs_are_valid(self):
        req = InferenceRequest(
            prompt="Hello world",
            model="test-model",
        )
        self.assertEqual(req.prompt, "Hello world")
        self.assertEqual(req.model, "test-model")
        self.assertEqual(req.generation.max_tokens, 256)
        self.assertEqual(req.server.port, 8000)
        self.assertEqual(req.server.protocol, ProtocolType.HTTP)

    def test_parse_valid_json(self):
        raw_json = json.dumps({
            "prompt": "Explain attention mechanism",
            "model": "vllm-llama3",
            "generation": {
                "max_tokens": 1024,
                "temperature": 0.5,
                "top_p": 0.8,
            },
            "server": {
                "host": "inference-host",
                "port": 9000,
                "protocol": "https",
            },
            "stream": True,
        })
        req = parse_config_json(raw_json)
        self.assertEqual(req.prompt, "Explain attention mechanism")
        self.assertEqual(req.model, "vllm-llama3")
        self.assertEqual(req.generation.max_tokens, 1024)
        self.assertEqual(req.generation.temperature, 0.5)
        self.assertEqual(req.generation.top_p, 0.8)
        self.assertEqual(req.server.port, 9000)
        self.assertEqual(req.server.protocol, ProtocolType.HTTPS)
        self.assertTrue(req.stream)

    def test_missing_prompt_raises_error(self):
        data = {"model": "llama3"}
        with self.assertRaises(ConfigValidationError) as ctx:
            parse_config_dict(data)
        self.assertIn("prompt", str(ctx.exception))

    def test_missing_model_raises_error(self):
        data = {"prompt": "Hello"}
        with self.assertRaises(ConfigValidationError) as ctx:
            parse_config_dict(data)
        self.assertIn("model", str(ctx.exception))

    def test_blank_prompt_raises_error(self):
        with self.assertRaises(ConfigValidationError):
            InferenceRequest(prompt="   ", model="llama3")

    def test_invalid_max_tokens_zero_or_negative(self):
        with self.assertRaises(ConfigValidationError):
            GenerationConfig(max_tokens=0)

        with self.assertRaises(ConfigValidationError):
            GenerationConfig(max_tokens=-10)

    def test_invalid_max_tokens_exceeds_bound(self):
        with self.assertRaises(ConfigValidationError):
            GenerationConfig(max_tokens=99999)

    def test_invalid_temperature_bounds(self):
        with self.assertRaises(ConfigValidationError):
            GenerationConfig(temperature=-0.1)

        with self.assertRaises(ConfigValidationError):
            GenerationConfig(temperature=2.5)

    def test_invalid_top_p_bounds(self):
        with self.assertRaises(ConfigValidationError):
            GenerationConfig(top_p=0.0)

        with self.assertRaises(ConfigValidationError):
            GenerationConfig(top_p=1.5)

    def test_invalid_server_port(self):
        with self.assertRaises(ConfigValidationError):
            ServerConfig(port=0)

        with self.assertRaises(ConfigValidationError):
            ServerConfig(port=70000)

    def test_invalid_protocol(self):
        with self.assertRaises(ConfigValidationError):
            ServerConfig(protocol="ftp")  # type: ignore

    def test_malformed_json_raises_validation_error(self):
        malformed = '{"prompt": "unclosed json'
        with self.assertRaises(ConfigValidationError) as ctx:
            parse_config_json(malformed)
        self.assertIn("Invalid JSON syntax", str(ctx.exception))

    def test_serialization_round_trip(self):
        original = InferenceRequest(
            prompt="Test prompt",
            model="test-model",
            generation=GenerationConfig(max_tokens=500, temperature=0.8),
            server=ServerConfig(host="localhost", port=8080, protocol=ProtocolType.HTTP),
            request_id="req-123",
            stream=True,
        )
        json_str = original.to_json()
        reconstructed = parse_config_json(json_str)
        self.assertEqual(reconstructed.prompt, original.prompt)
        self.assertEqual(reconstructed.model, original.model)
        self.assertEqual(reconstructed.generation.max_tokens, 500)
        self.assertEqual(reconstructed.generation.temperature, 0.8)
        self.assertEqual(reconstructed.server.port, 8080)
        self.assertEqual(reconstructed.request_id, "req-123")
        self.assertTrue(reconstructed.stream)

    def test_load_from_file(self):
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as tmp:
            tmp.write(json.dumps({
                "prompt": "Test from file",
                "model": "file-model",
                "generation": {"max_tokens": 128},
            }))
            tmp_path = Path(tmp.name)

        try:
            req = load_config_from_file(tmp_path)
            self.assertEqual(req.prompt, "Test from file")
            self.assertEqual(req.model, "file-model")
            self.assertEqual(req.generation.max_tokens, 128)
        finally:
            tmp_path.unlink()


class TestCli(unittest.TestCase):
    """Test suite for the infer-config CLI tool."""

    def test_cli_sample_generation(self):
        exit_code = main(["--sample"])
        self.assertEqual(exit_code, 0)

    def test_cli_valid_file(self):
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as tmp:
            tmp.write(json.dumps({
                "prompt": "CLI test",
                "model": "cli-model",
            }))
            tmp_path = Path(tmp.name)

        try:
            exit_code = main(["--config", str(tmp_path), "--validate-only"])
            self.assertEqual(exit_code, 0)
        finally:
            tmp_path.unlink()

    def test_cli_invalid_file(self):
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as tmp:
            tmp.write(json.dumps({
                "prompt": "CLI test without model",
            }))
            tmp_path = Path(tmp.name)

        try:
            exit_code = main([str(tmp_path)])
            self.assertEqual(exit_code, 1)
        finally:
            tmp_path.unlink()

    def test_cli_missing_file(self):
        exit_code = main(["non_existent_file.json"])
        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
