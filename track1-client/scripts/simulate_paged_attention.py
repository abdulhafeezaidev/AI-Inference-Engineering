import json

def simulate_memory_systems():
    print("--- Memory Allocation Simulator: Static vs PagedAttention ---")
    
    # 10 Users request different prompt/generation lengths (in tokens)
    user_requests = [10, 50, 200, 15, 1000, 30, 500, 12, 100, 80]
    total_users = len(user_requests)
    
    # ---------------------------------------------------------
    # 1. STATIC ALLOCATION (The old way)
    # Every user gets the max context window reserved contiguously
    # ---------------------------------------------------------
    max_context = 2048
    
    static_total_reserved = total_users * max_context
    static_actual_used = sum(user_requests)
    static_wasted = static_total_reserved - static_actual_used
    static_waste_percent = (static_wasted / static_total_reserved) * 100
    
    print("\n[1] STATIC ALLOCATION (Pre-allocating Max Context):")
    print(f"Total Tokens Reserved : {static_total_reserved:,}")
    print(f"Actual Tokens Used    : {static_actual_used:,}")
    print(f"Tokens Wasted         : {static_wasted:,} ({static_waste_percent:.2f}% Internal Fragmentation)")
    
    # ---------------------------------------------------------
    # 2. PAGED ATTENTION (The vLLM way)
    # Memory is divided into small fixed-size blocks
    # ---------------------------------------------------------
    block_size = 16
    
    paged_total_reserved = 0
    
    for tokens in user_requests:
        # Calculate how many blocks this user needs
        # Ceiling division: (tokens + block_size - 1) // block_size
        blocks_needed = (tokens + block_size - 1) // block_size
        tokens_reserved_for_user = blocks_needed * block_size
        paged_total_reserved += tokens_reserved_for_user
        
    paged_actual_used = sum(user_requests)
    paged_wasted = paged_total_reserved - paged_actual_used
    paged_waste_percent = (paged_wasted / paged_total_reserved) * 100
    
    print(f"\n[2] PAGED ATTENTION (Block Size = {block_size}):")
    print(f"Total Tokens Reserved : {paged_total_reserved:,}")
    print(f"Actual Tokens Used    : {paged_actual_used:,}")
    print(f"Tokens Wasted         : {paged_wasted:,} ({paged_waste_percent:.2f}% Internal Fragmentation)")
    
    # --- Conclusion ---
    savings_factor = static_wasted / max(paged_wasted, 1)
    print(f"\n-> PagedAttention wasted {savings_factor:.1f}x LESS memory than Static Allocation!")
    
    # Save log
    log_data = {
        "static_waste_percent": static_waste_percent,
        "paged_waste_percent": paged_waste_percent,
        "savings_factor": savings_factor
    }
    with open("artifacts/day29/paged_attention_sim.json", "w") as f:
        json.dump(log_data, f, indent=4)
    print("\nLogs saved to artifacts/day29/paged_attention_sim.json")

if __name__ == "__main__":
    simulate_memory_systems()
