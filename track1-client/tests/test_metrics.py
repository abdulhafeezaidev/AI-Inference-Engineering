import pytest
from inference_client.metrics import (
    StreamMetricsRecord,
    compute_stream_metrics,
    summarize_measured_records,
)


def test_hand_computed_three_token_trace():
    """
    Hand-computed trace:
      t_start = 0.000s
      tok_1   = 0.100s -> TTFT = 100.0 ms
      tok_2   = 0.130s -> delta_1 = 30.0 ms
      tok_3   = 0.170s -> delta_2 = 40.0 ms
      t_end   = 0.170s -> total = 170.0 ms, mean ITL = 35.0 ms
      output_tokens_per_sec = 3 / 0.170 = 17.647 tok/s
      decode_tokens_per_sec = 2 / 0.070 = 28.571 tok/s
    """
    rec = compute_stream_metrics(
        request_id="req-3tok",
        phase="measured",
        t_start_sec=0.000,
        token_timestamps_sec=[0.100, 0.130, 0.170],
        t_end_sec=0.170,
        prompt_tokens=12,
        output_tokens=3,
    )

    assert rec.ttft_ms == 100.0
    assert rec.token_deltas_ms == [30.0, 40.0]
    assert rec.mean_itl_ms == 35.0
    assert rec.total_latency_ms == 170.0
    assert rec.prompt_tokens == 12
    assert rec.output_tokens == 3
    assert rec.output_tokens_per_sec == pytest.approx(17.647, abs=1e-3)
    assert rec.decode_tokens_per_sec == pytest.approx(28.571, abs=1e-3)


def test_single_token_stream_edge_case():
    rec = compute_stream_metrics(
        request_id="req-1tok",
        phase="measured",
        t_start_sec=1.000,
        token_timestamps_sec=[1.080],
        t_end_sec=1.085,
        prompt_tokens=5,
        output_tokens=1,
    )
    assert rec.ttft_ms == 80.0
    assert rec.token_deltas_ms == []
    assert rec.mean_itl_ms == 0.0
    assert rec.total_latency_ms == 85.0
    assert rec.decode_tokens_per_sec == 0.0


def test_warmup_and_error_isolation_in_summary():
    warmup = compute_stream_metrics("w-0", "warmup", 0.0, [0.500, 0.600], 0.600, 10, 2)
    ok_run = compute_stream_metrics("m-0", "measured", 0.0, [0.100, 0.130, 0.170], 0.170, 12, 3)
    err_run = StreamMetricsRecord(
        request_id="m-1",
        phase="measured",
        status="error",
        prompt_tokens=None,
        output_tokens=None,
        ttft_ms=None,
        token_deltas_ms=[],
        mean_itl_ms=None,
        total_latency_ms=12.5,
        output_tokens_per_sec=None,
        decode_tokens_per_sec=None,
        error_type="HTTPError_503",
        error_message="Engine overloaded",
    )

    summary = summarize_measured_records([warmup, ok_run, err_run])
    assert summary["measured_total"] == 2
    assert summary["measured_ok"] == 1
    assert summary["measured_errors"] == 1
    assert summary["failure_rate"] == 0.5
    assert summary["mean_ttft_ms"] == 100.0
    assert summary["mean_itl_ms"] == 35.0
