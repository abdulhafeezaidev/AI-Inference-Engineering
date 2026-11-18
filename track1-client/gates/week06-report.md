# Week 6 Gate Report: Inference Benchmarking & Statistical Hygiene

## 1. Benchmark Reproducibility & Provenance
- **Reproduction Command:** `./.venv-linux/bin/python scripts/run_week06_gate.py`
- **Artifacts:**
  - `artifacts/day41/week06-benchmark.json` (Full run manifest, sweep summaries, independent verification, outlier & failure test)
  - `artifacts/day41/week06-sweep.csv` (Concurrency sweep metrics across `c = 1, 2, 4, 8`)
  - `artifacts/day41/week06-latency-throughput.svg` (`p50` and `p95` E2E Latency vs. System Throughput curve)
- **Statistical Hygiene Controls:** Fixed model (`HuggingFaceTB/SmolLM-135M-Instruct`), fixed prompt token length (`12`), fixed output token cap (`8`), `2` warm-up requests excluded prior to `6` measured repetitions per concurrency level, `nearest-rank` percentiles without outlier trimming, and separate accounting for failed requests.

## 2. Independent Metric Verification & Outlier Analysis
- **Independent Calculation Match (`c1-m0`):**
  - `TTFT`: `30.77 ms` (Harness) == `30.77 ms` (Independent)
  - `Mean ITL`: `7.583 ms` (Harness) == `7.583 ms` (Independent)
  - `Total E2E Latency`: `84.316 ms` (Harness) == `84.316 ms` (Independent)
  - `Output Throughput`: `94.881 tok/s` (Harness) == `94.881 tok/s` (Independent)
- **Tail Outlier Identified (`c4-m3`):** Request `c4-m3` recorded `274.97 ms` total latency (`180.388 ms` TTFT), representing a **2.73x slowdown** over the global `p50` E2E latency (`100.796 ms`), preserved intact in `p95` tail reporting.

## 3. Bounded Assumption Break & Failure Diagnosis
- **Broken Assumption:** Reduced engine admission capacity to `max_inflight=1` under a `concurrency=4` burst.
- **Observed Symptom:** `83.33%` (`5/6`) requests failed immediately with `HTTPError_503` (capacity exceeded) while the single admitted request completed cleanly.
- **Recovery:** Restoring `max_inflight=12` returned the failure rate to `0.0%` (`0/6` errors).

## 4. Five-Minute Engineering Review Defense
1. **Why Warm-Up Matters:** Initial inference requests trigger cold JIT/kernel compilation, allocator slab growth, and connection pool establishment. Excluding warm-up requests prevents one-time startup overhead from distorting steady-state `p50`/`p95` latency distributions.
2. **Why `p95` (Nearest-Rank) Matters Over Mean Alone:** Mean and `p50` mask queueing spikes and KV-cache contention experienced by real users under load. Nearest-rank `p95` guarantees that actual worst-case user experiences are surfaced without artificial interpolation smoothing.
3. **Why ITL (Inter-Token Latency) Matters Separately from TTFT:** TTFT measures compute-bound prompt prefill + queueing delay before the first token appears, whereas mean ITL measures memory-bandwidth-bound autoregressive decode cadence. Separating them pinpoints whether a bottleneck is prefill/queueing (`TTFT` spike) or decode memory pressure (`ITL` rise).
4. **Why Failure Rate Must Be Reported Separately from Latency:** Fast-failing errors (e.g., `HTTP 503` in `<5 ms`) would artificially lower latency percentiles if mixed with successful requests. Isolating errors in `failure_rate` ensures reliability regressions are never disguised as speedups.
5. **Rejecting an Unfair Benchmark Claim:**
   > *Claim:* "Engine X achieves 250 tok/s and 12 ms latency—3x faster than Engine Y."
   > *Technical Rejection:* Rejected because the claim omits (a) concurrency level (mixing single-user latency with multi-user aggregate throughput), (b) prompt/output token lengths (short outputs inflate apparent throughput or hide KV-cache growth), (c) warm-up exclusion policy, (d) `p95`/`p99` tail latency and `failure_rate` (an overloaded server dropping 50% of requests via 503 errors can appear artificially fast on survivors), and (e) environmental A/A noise variance baselines.
