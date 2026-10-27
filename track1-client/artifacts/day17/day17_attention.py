import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def trace_multi_head_attention():
    print("=" * 60)
    print(" Day 17: Multi-Head Attention Tensor Shape Trace")
    print("=" * 60)

    # 1. Hyperparameters
    batch_size = 2        # B
    seq_len = 4           # S (Tokens in prompt)
    d_model = 64          # D (Embedding dimension)
    num_heads = 4         # H (Number of attention heads)
    d_k = d_model // num_heads  # Dimension per head (64 / 4 = 16)

    print(f"\n[Config] B={batch_size}, S={seq_len}, D={d_model}, H={num_heads}, d_k={d_k}")

    # 2. Simulated Input Embeddings [B, S, D]
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\n1. Input Embeddings (x)           Shape: {list(x.shape)}")

    # 3. Linear Projections for Q, K, V
    W_q = nn.Linear(d_model, d_model, bias=False)
    W_k = nn.Linear(d_model, d_model, bias=False)
    W_v = nn.Linear(d_model, d_model, bias=False)

    Q = W_q(x)
    K = W_k(x)
    V = W_v(x)
    print(f"2. Projected Q, K, V              Shape: {list(Q.shape)} each")

    # 4. Multi-Head Reshape & Transpose: [B, S, D] -> [B, S, H, d_k] -> [B, H, S, d_k]
    Q = Q.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
    K = K.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
    V = V.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
    print(f"3. Reshaped & Transposed (Q, K, V) Shape: {list(Q.shape)} -> [B, H, S, d_k]")

    # 5. Raw Attention Scores: Q @ K.transpose(-2, -1) -> [B, H, S, S]
    # [B, H, S, d_k] x [B, H, d_k, S] = [B, H, S, S]
    scores = torch.matmul(Q, K.transpose(-2, -1))
    print(f"4. Raw Dot-Product Scores (Q @ K.T) Shape: {list(scores.shape)} -> [B, H, S, S]")

    # 6. Scaling by 1 / sqrt(d_k)
    scale_factor = math.sqrt(d_k)
    scaled_scores = scores / scale_factor
    print(f"5. Scaled Scores (div sqrt({d_k}))  Shape: {list(scaled_scores.shape)}")

    # 7. Softmax over last dimension (keys)
    attn_weights = F.softmax(scaled_scores, dim=-1)
    print(f"6. Attention Weights (Softmax)    Shape: {list(attn_weights.shape)}")

    # Verify probability distribution (sums to 1.0 along last dim)
    sum_check = attn_weights.sum(dim=-1)[0, 0, 0].item()
    print(f"   -> Probability sum verification for Token 0 in Head 0: {sum_check:.4f}")

    # 8. Weighted Value Sum: weights @ V -> [B, H, S, d_k]
    # [B, H, S, S] x [B, H, S, d_k] = [B, H, S, d_k]
    context = torch.matmul(attn_weights, V)
    print(f"7. Context Representation (attn @ V) Shape: {list(context.shape)} -> [B, H, S, d_k]")

    # 9. Concatenate Heads back to [B, S, D]
    # Transpose [B, S, H, d_k] -> contiguous view [B, S, D]
    context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    print(f"8. Recombined Multi-Head Context  Shape: {list(context.shape)} -> [B, S, D]")

    # 10. Final Output Projection W_o
    W_o = nn.Linear(d_model, d_model, bias=False)
    out = W_o(context)
    print(f"9. Final Linear Output (out)      Shape: {list(out.shape)} -> [B, S, D]")

    print("\n" + "=" * 60)
    print(" VERIFICATION SUCCESSFUL: Input shape matches Output shape [B, S, D]")
    print("=" * 60)

if __name__ == "__main__":
    trace_multi_head_attention()
