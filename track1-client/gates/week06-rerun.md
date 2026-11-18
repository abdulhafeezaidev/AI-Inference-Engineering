# Week 6 Benchmark Gate Rerun Report (Day 42)

- **Generated (UTC):** `2026-11-18T12:03:44.265396+00:00`
- **Git Commit:** `86a2ebd`
- **Reproduce Command:** `./.venv-linux/bin/python scripts/run_day42_rerun.py`
- **Gate Verdict:** `PASS — Reproducible across clean cold restart; ready for Week 7 profiling`

## 1. Run Manifest & Statistical Hygiene Controls

- **Hardware:** `x86_64`, `12` CPU cores (`CPU-only WSL2 harness (no local NVIDIA GPU)`)
- **Software:** `Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43`, Python `3.14.4`, `GateBenchmarkServer (ThreadingHTTPServer SSE mock)`
- **Server Config:** `model_id=synthetic-sse-token-stream-v1`, `max_slots=4`, `max_inflight=12`, `inject_outlier_every=11`
- **Clean Restart Verification:** Initial server stopped and port verified released before cold restart (`restart_cycle_ms=423.599 ms`)
- **Workload Controls:** `levels=[1, 2, 4, 8]`, `warmup_excluded=8` (2/level), `measured_total=24` (6/level), `prompt_tokens=12`, `max_tokens=8`, `measured_failures=0`

## 2. Clean-Restart Sweep Comparison (Day 42 vs. Day 41 Baseline)

| Concurrency | OK/N | Fail Rate | System TPS (Δ%) | TTFT p50 ms (Δ%) | TTFT p95 ms (Δ%) | ITL p50 ms (Δ%) | E2E p50 ms (Δ%) | E2E p95 ms (Δ%) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6/6 | 0.0 | 99.30 (+3.91%) | 27.500 (-4.07%) | 28.342 (-7.89%) | 7.413 (-2.24%) | 79.635 (-3.95%) | 80.856 (-4.10%) |
| 2 | 6/6 | 0.0 | 121.85 (+2.46%) | 30.842 (-5.24%) | 168.689 (-0.47%) | 9.333 (-1.64%) | 97.506 (-1.64%) | 221.998 (-0.69%) |
| 4 | 6/6 | 0.0 | 170.44 (+0.25%) | 36.477 (-2.18%) | 180.190 (-0.11%) | 11.388 (-1.70%) | 119.083 (-0.47%) | 274.157 (-0.30%) |
| 8 | 6/6 | 0.0 | 223.10 (+1.91%) | 42.425 (-1.53%) | 115.985 (+1.67%) | 11.350 (-0.96%) | 123.029 (-0.61%) | 209.622 (+0.31%) |

## 3. Bounded Assumption Failure & Recovery Test

- **Broken Assumption:** Set client read timeout=0.015s (15ms) below minimum server TTFT (~25ms)
- **Observed Symptom:** `TimeoutError` (`failure_rate=1.0`)
- **Restored State:** Restored client timeout=5.0s (`failure_rate=0.0`)
