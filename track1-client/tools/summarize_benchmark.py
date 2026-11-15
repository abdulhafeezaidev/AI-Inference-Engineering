#!/usr/bin/env python3
import argparse, json, math, os, platform, statistics, sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

PERCENTILE_CONVENTION = {
    "method": "nearest-rank",
    "formula": "sorted_values[min(n - 1, max(0, ceil((q / 100.0) * n) - 1))]",
    "outlier_policy": "strict_preservation_no_trimming_or_winsorization",
    "failure_handling": "excluded_from_latency_percentiles_reported_in_failure_rate",
}

def nearest_rank_percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    idx = min(n - 1, max(0, math.ceil((q / 100.0) * n) - 1))
    return round(float(ordered[idx]), 3)

def summarize_distribution(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p95": None, "p99": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": round(float(min(values)), 3),
        "p50": nearest_rank_percentile(values, 50),
        "p95": nearest_rank_percentile(values, 95),
        "p99": nearest_rank_percentile(values, 99),
        "max": round(float(max(values)), 3),
        "mean": round(float(statistics.mean(values)), 3),
    }

def summarize_records(records: List[Dict[str, Any]], source_file: str = "in-memory") -> Dict[str, Any]:
    warmup = [r for r in records if r.get("phase") == "warmup"]
    measured = [r for r in records if r.get("phase") == "measured"]
    if not measured:
        raise ValueError("No measured records found")
    ok = [r for r in measured if r.get("status") == "ok"]
    err = [r for r in measured if r.get("status") != "ok"]
    fail_rate = round(len(err) / len(measured), 4)

    ttfts = [float(r["ttft_ms"]) for r in ok if r.get("ttft_ms") is not None]
    itl_key = "mean_itl_ms" if any("mean_itl_ms" in r for r in ok) else "tpot_ms"
    itls = [float(r[itl_key]) for r in ok if r.get(itl_key) is not None]
    lats = [float(r["total_latency_ms"]) for r in ok if r.get("total_latency_ms") is not None]
    tps_key = "output_tokens_per_sec" if any("output_tokens_per_sec" in r for r in ok) else "tokens_per_sec"
    tpss = [float(r[tps_key]) for r in ok if r.get(tps_key) is not None]

    tot_tok = sum(int(r.get("output_tokens") or r.get("completion_tokens") or 0) for r in ok)
    tot_sec = sum(float(r.get("total_latency_ms") or 0.0) for r in ok) / 1000.0

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": source_file,
        "environment": {"os": platform.platform(), "python_version": sys.version.split()[0], "cpu_cores": os.cpu_count()},
        "percentile_convention": PERCENTILE_CONVENTION,
        "sample_counts": {
            "total_jsonl_rows": len(records), "warmup_excluded": len(warmup),
            "measured_total": len(measured), "measured_ok": len(ok), "measured_error": len(err),
        },
        "reliability": {
            "failure_rate": fail_rate, "success_rate": round(1.0 - fail_rate, 4),
            "error_breakdown": dict(Counter(r.get("error_type") or "unknown" for r in err)),
        },
        "throughput": {
            "total_output_tokens": tot_tok,
            "aggregate_output_tokens_per_sec": round(tot_tok / tot_sec, 3) if tot_sec > 0 else None,
            "per_request_tps_distribution": summarize_distribution(tpss),
        },
        "latency_ms": {
            "ttft_ms": summarize_distribution(ttfts),
            "itl_ms": summarize_distribution(itls),
            "total_latency_ms": summarize_distribution(lats),
        },
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="artifacts/day37/day37-latency.jsonl")
    p.add_argument("--output", default="artifacts/day38/day38-summary.json")
    args = p.parse_args()
    with open(args.input, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    summary = summarize_records(records, source_file=args.input)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
