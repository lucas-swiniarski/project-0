import torch
import torch.nn.functional as F
import time
from tokenizers import Tokenizer

def sample_next_token(logits: torch.Tensor, 
                      top_k: int = 0, 
                      top_p: float = 0.0,
                      temperature: float = 1.0) -> torch.Tensor:
    """Sample next word from logits.

    Args:
        logits (torch.Tensor): Logits. Shape (B, T, vocab_size).
        top_k (int, optional): Keep top-k logits. Defaults to 0.
        top_p (float, optional): Nucleus p. Keep n tokens per batch elem such that sum(prob) >= top_p. Defaults to 0.0.
        temperature (float, optional): Sampling temperature. Defaults to 1.0.

    Returns:
        idx: Token indices. Shape (B, T+1).
    """
    B, T, vocab_size = logits.shape
    logits = logits[:, -1, :] / temperature # (B, vocab_size)
                
    if top_k > 0:
        top_k_v, top_k_idx = torch.topk(logits, k=top_k, dim=-1) # (B, top_k)
        logits[logits < top_k_v[:, -1].unsqueeze(-1)] = float('-inf')
        
    probs = F.softmax(logits, dim=-1) # (B, vocab_size)

    if top_p > 0:
        sorted_probs_v, sorted_probs_idx = torch.sort(probs, descending=True, dim=-1) # (B, vocab_size)
        sup_top_p_mask = torch.cumsum(sorted_probs_v, dim=-1) > top_p # True when cumsum > top_p (B, vocab_size)
        sup_top_p_mask = ~torch.concat([
            torch.zeros((B, 1), dtype=torch.bool, device=probs.device), 
            sup_top_p_mask[:, :-1]], axis=-1) # True until first cumsum > top_p, (B, vocab_size)
        # Add a small epsilon to prevent division by zero if all probabilities are filtered out
        probs = torch.zeros_like(probs).scatter_(1, sorted_probs_idx, sup_top_p_mask * sorted_probs_v) + 1e-9
        probs = probs / probs.sum(-1, keepdim=True)
    
    next_idx = torch.multinomial(probs, num_samples=1)
    log_prob = torch.log(torch.gather(probs, 1, next_idx))
    return next_idx, log_prob

@torch.no_grad()
def estimate_loss(model, data_loader, eval_batches):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_batches)
        for k in range(eval_batches):
            X, Y = data_loader.get_batch(split)
            logits, loss, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

@torch.no_grad()
def generate_text(model, tokenizer, data_loader, context_size, max_new_tokens, top_k, top_p, temperature):
    model.eval()
    # --- Generation Comparison ---
    print('\n--- Generation Comparison ---')
    # 1. Load a batch from validation
    X, _ = data_loader.get_batch('val')
    # 2. Keep the first context_size / 2 tokens of the first 2 elements
    context = X[:2, :context_size // 2]

    # 3. Generate with caching (default)
    print("\nGenerating with caching...")
    torch.manual_seed(0) # Set seed for reproducibility of sampling
    start_time = time.time()
    generated_tokens_cached, _ = model.generate(context, max_new_tokens=max_new_tokens, top_k=top_k, top_p=top_p, temperature=temperature, use_cache=True)
    duration_cached = time.time() - start_time
    print(f"Time taken: {duration_cached:.4f} seconds")

    # 4. Generate without caching
    print("\nGenerating without caching...")
    torch.manual_seed(0) # Reset seed for reproducibility
    start_time = time.time()
    generated_tokens, _ = model.generate(context, max_new_tokens=max_new_tokens, top_k=top_k, top_p=top_p, temperature=temperature, use_cache=False)
    duration = time.time() - start_time
    print(f"Time taken: {duration:.4f} seconds")

    context_len = context.shape[1]
    for i in range(len(generated_tokens)):
        print(f"\n--- Sample {i+1} ---")
        context_text = tokenizer.decode(context[i].tolist())
        generated_text = tokenizer.decode(generated_tokens[i, context_len:].tolist())
        print(f"  Without cache:")
        print(f"    Context:   '{context_text}'")
        print(f"    Generated: '{generated_text}'")
        generated_text_cached = tokenizer.decode(generated_tokens_cached[i, context_len:].tolist())
        print(f"  With cache:\n    Generated: '{generated_text_cached}'")
    print('--- End Generation Comparison ---\n')
    model.train()