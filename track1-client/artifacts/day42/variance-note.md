# Day 42 Benchmark Variance & Statistical Hygiene Note

- **Variance Threshold:** `±10.0%` across all sweep metrics (`system_tps`, `ttft_p50_ms`, `ttft_p95_ms`, `itl_p50_ms`, `e2e_p50_ms`, `e2e_p95_ms`).
- **Metrics Exceeding Threshold:** `0` (none).

## Detailed Delta Breakdown & Classification

1. **Median & Throughput Jitter (`< ±5.5%` across all levels):**
   - At `c=2`, `ttft_p50_ms` shifted from `32.547 ms` (Day 41) to `30.842 ms` (`-5.24%`), and `system_tps` shifted from `118.93` to `121.85` (`+2.46%`).
   - At `c=8`, `system_tps` shifted from `218.91` to `223.10` (`+1.91%`), and `ttft_p50_ms` shifted from `43.086 ms` to `42.425 ms` (`-1.53%`).
   - **Classification:** `Expected OS/WSL2 thread scheduling jitter` — well inside the `±10%` repeatability envelope.

2. **Tail Percentile Stability (`ttft_p95_ms` & `e2e_p95_ms`):**
   - Because the cold server restart reset request sequence counters cleanly, the periodic `140 ms` tail outlier (`inject_outlier_every=11`) aligned identically on `c=2` (`168.689 ms` vs `169.482 ms`, `-0.47%`) and `c=4` (`180.190 ms` vs `180.388 ms`, `-0.11%`).
   - **Classification:** `Expected deterministic tail preservation` — confirms warm-up exclusion and percentile definitions survived the restart intact.
