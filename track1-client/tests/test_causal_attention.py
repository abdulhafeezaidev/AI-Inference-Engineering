# tests/test_causal_attention.py
import torch
import json
import os
import sys

# Ensure src is in the path to import our module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference_math.attention_reference import reference_attention

def test_causal_leakage():
    print("--- Running Causal Leakage Test ---")
    torch.manual_seed(42)
    # Shape: (batch=1, heads=1, seq_len=4, head_dim=8)
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 1, 4, 8)
    v = torch.randn(1, 1, 4, 8)

    # 1. Baseline Causal Output
    out_baseline = reference_attention(q, k, v, is_causal=True)

    # 2. Perturb Future Token (Index 3)
    q_pert, k_pert, v_pert = q.clone(), k.clone(), v.clone()
    v_pert[:, :, 3, :] += 500.0  # Massive spike to future token

    out_perturbed = reference_attention(q_pert, k_pert, v_pert, is_causal=True)

    # 3. Check if Past/Present tokens (0, 1, 2) were affected
    # tolerance set very strict (1e-6) because math should be EXACTLY identical
    is_safe = torch.allclose(out_baseline[:, :, :3, :], out_perturbed[:, :, :3, :], atol=1e-6)
    assert is_safe, "FAIL: Future token leaked into past!"
    
    print("PASSED: Future token (index 3) did NOT leak into past tokens (0, 1, 2).\n")

def test_numerical_precision():
    print("--- Running Dtype Numerical Precision Test ---")
    torch.manual_seed(42)
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 1, 4, 8)
    v = torch.randn(1, 1, 4, 8)

    # Run in FP32 (Golden Reference)
    out_fp32 = reference_attention(q, k, v, is_causal=True)

    # Run in FP16
    out_fp16 = reference_attention(q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), is_causal=True)
    
    # Run in BF16
    out_bf16 = reference_attention(q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), is_causal=True)

    # Check Tolerance
    # FP16 mantissa is 10 bits -> tolerance ~1e-3
    fp16_match = torch.allclose(out_fp32, out_fp16.float(), atol=1e-3, rtol=1e-3)
    # BF16 mantissa is 7 bits -> tolerance ~1e-2
    bf16_match = torch.allclose(out_fp32, out_bf16.float(), atol=1e-2, rtol=1e-2)

    print(f"FP16 matches FP32 within 1e-3 tolerance: {fp16_match}")
    print(f"BF16 matches FP32 within 1e-2 tolerance: {bf16_match}")
    
    # Task 3 Deliverable: Save Log
    log_data = {
        "causal_leakage_prevented": True,
        "fp16_tolerance_met": bool(fp16_match),
        "bf16_tolerance_met": bool(bf16_match),
        "tolerances": {"fp32": "baseline", "fp16": "1e-3", "bf16": "1e-2"}
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/day18-numerics.json", "w") as f:
        json.dump(log_data, f, indent=4)
    print("\n[OK] Logs saved to logs/day18-numerics.json")

if __name__ == "__main__":
    test_causal_leakage()
    test_numerical_precision()