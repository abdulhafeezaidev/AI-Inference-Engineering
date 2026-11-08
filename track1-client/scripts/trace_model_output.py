import torch
import time
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

def run_model_trace():
    print("Loading tokenizer and model (SmolLM-135M)...")
    model_name = "HuggingFaceTB/SmolLM-135M"
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    prompt = "The most important rule of inference engineering is"
    print(f"\nPrompt: '{prompt}'")
    
    # 1. Tokenize Input
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"Input IDs shape: {input_ids.shape}")
    
    # 2. Forward Pass (Trace)
    print("\nRunning forward pass...")
    start_time = time.perf_counter()
    
    with torch.no_grad():
        # output_hidden_states=True enables us to see the internal layers
        outputs = model(input_ids, output_hidden_states=True)
        
    end_time = time.perf_counter()
    
    # 3. Extract Shapes
    logits = outputs.logits
    # hidden_states is a tuple containing the output of embedding layer + all transformer layers
    hidden_states = outputs.hidden_states
    
    # The last layer's hidden state before the language modeling head
    last_hidden_state = hidden_states[-1] 
    
    print(f"\nLogits shape: {logits.shape} (batch, seq_len, vocab_size)")
    print(f"Last Hidden State shape: {last_hidden_state.shape} (batch, seq_len, hidden_dim)")
    
    # 4. Generate next token (Greedy)
    # We take the logits of the *last* token in the sequence to predict the next word
    next_token_logits = logits[0, -1, :] 
    next_token_id = torch.argmax(next_token_logits, dim=-1).item()
    next_token_text = tokenizer.decode(next_token_id)
    
    print(f"\nGenerated Next Token ID: {next_token_id}")
    print(f"Generated Next Token Text: '{next_token_text}'")
    print(f"Forward Pass Timing: {(end_time - start_time) * 1000:.2f} ms")
    
    # Save Trace Data
    trace_data = {
        "prompt": prompt,
        "input_ids_shape": list(input_ids.shape),
        "logits_shape": list(logits.shape),
        "last_hidden_state_shape": list(last_hidden_state.shape),
        "next_token_id": next_token_id,
        "next_token_text": next_token_text,
        "forward_pass_ms": (end_time - start_time) * 1000
    }
    
    with open("artifacts/day20/day20-model-output.json", "w") as f:
        json.dump(trace_data, f, indent=4)
        
    print("\nTrace saved to artifacts/day20/day20-model-output.json")

if __name__ == "__main__":
    run_model_trace()
