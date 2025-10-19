import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Tuple
from tokenizers import Tokenizer

if False: # TYPE_CHECKING
    from dataset.pre_training.data_loader import DataLoader
    from transformer.model import MyTransformer

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
        A tuple of (next_idx, log_prob) where next_idx is the sampled token index and log_prob is its log probability.
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
    return next_idx, log_prob.cpu()

@torch.no_grad()
def estimate_cross_entropy_loss(
    model: 'MyTransformer', 
    data_loader: 'DataLoader', 
    eval_batches: int) -> dict[str, float]:
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_batches)
        for k in range(eval_batches):
            X, Y = data_loader.get_batch(split) # (B, T)
            B, T = X.shape
            mask = torch.tril(torch.ones(T, T, device=X.device))
            logits, loss, _ = model(X, Y, mask=mask)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

@torch.no_grad()
def estimate_dpo_loss(
    pi_theta: 'MyTransformer',
    pi_ref: 'MyTransformer',
    data_loader: 'DataLoader', 
    eval_batches: int) -> dict[str, float]:
    pass

def dpo_loss(
    pi_theta: 'MyTransformer',
    pi_ref: 'MyTransformer',
    x: torch.Tensor,
    y: torch.Tensor,
    r: torch.Tensor
) -> torch.Tensor:
    pass

@torch.no_grad()
def generate_text(model: 'MyTransformer',
                  tokenizer: Tokenizer,
                  data_loader: 'DataLoader',
                  context_size: int,
                  max_new_tokens: int,
                  stop_token: int | None = None,
                  top_k: int = 64,
                  top_p: float = 0.9,
                  temperature: float = 1.0, 
                  also_no_cache_decode: bool = False) -> str:
    # Save the current RNG state to isolate generation from training randomness
    rng_state = torch.get_rng_state()
    model.eval()
    
    # 1. Load a test data point
    # get_batch(test) return (1, input_seq_len)
    X, _ = data_loader.get_batch('test')

    # 2. Generate with caching
    print("\n--- Generating Text (with cache) ---")
    start_time = time.time()
    generated_tokens_cached, log_probs_cached = model.generate(X, 
                                                               max_new_tokens=max_new_tokens,
                                                               stop_token=stop_token,
                                                               top_k=top_k, 
                                                               top_p=top_p,
                                                               temperature=temperature,
                                                               use_cache=True)
    duration_cached = time.time() - start_time

    context_len = X.shape[1]
    print(f"\n--- Sample ---")
    context_text = tokenizer.decode(X[0].tolist(), skip_special_tokens=False)
    generated_text_cached = tokenizer.decode(generated_tokens_cached[0, context_len:].tolist(), skip_special_tokens=False)
    print(f"  Context:   '{context_text}'")
    print(f"  Generated: '{generated_text_cached}'")
    print(f'Time to generate: {duration_cached:.4f}s')

    # Format for logging
    log_text = f"**Context:**\n\n```\n{context_text}\n```\n\n**Generated:**\n\n```\n{generated_text_cached}\n```"

    if also_no_cache_decode:
        print("\n--- Comparing with non-cached generation ---")
        start_time = time.time()
        generated_tokens, log_probs = model.generate(X, 
                                                     max_new_tokens=max_new_tokens,
                                                     stop_token=stop_token,
                                                     top_k=top_k, 
                                                     top_p=top_p, 
                                                     temperature=temperature,
                                                     use_cache=False)
        duration = time.time() - start_time
        
        generated_text = tokenizer.decode(generated_tokens[0, context_len:].tolist(), skip_special_tokens=False)
        print(f"  Generated (no cache): '{generated_text}'")
        print(f'Time to generate - cache: {duration_cached:.4f}s, no cache: {duration:.4f}s')
        print(f'log-probs - cache: {log_probs_cached.mean().item():.4f}, no cache: {log_probs.mean().item():.4f}')

    print('--- End Generation ---\n')
    # Restore the original RNG state
    torch.set_rng_state(rng_state)
    model.train()
    return log_text

def expand_model_vocabulary(state_dict, old_vocab_size, new_vocab_size):
    """
    Resizes the token embedding and output projection layers of a model's state_dict
    to accommodate a new vocabulary size.

    Args:
        state_dict (dict): The model's state dictionary.
        old_vocab_size (int): The original vocabulary size.
        new_vocab_size (int): The target vocabulary size.

    Returns:
        dict: The modified state dictionary with resized layers.
    """
    if old_vocab_size == new_vocab_size:
        return state_dict

    print(f"Vocabulary size mismatch. Resizing model embeddings from {old_vocab_size} to {new_vocab_size}.")

    # 1. Resize token embedding table
    old_emb = state_dict['token_embedding.token_embedding_table.weight']
    new_emb = torch.zeros(new_vocab_size, old_emb.size(1), dtype=old_emb.dtype, device=old_emb.device)
    nn.init.normal_(new_emb, mean=0.0, std=0.02)
    new_emb[:old_vocab_size, :] = old_emb
    state_dict['token_embedding.token_embedding_table.weight'] = new_emb

    # 2. Resize output projection layer
    old_proj = state_dict['W_o.weight']
    new_proj = torch.zeros(new_vocab_size, old_proj.size(1), dtype=old_proj.dtype, device=old_proj.device)
    nn.init.normal_(new_proj, mean=0.0, std=0.02)
    new_proj[:old_vocab_size, :] = old_proj
    state_dict['W_o.weight'] = new_proj

    if 'W_o.bias' in state_dict:
        old_bias = state_dict['W_o.bias']
        new_bias = torch.zeros(new_vocab_size, dtype=old_bias.dtype, device=old_bias.device)
        new_bias[:old_vocab_size] = old_bias
        state_dict['W_o.bias'] = new_bias

    return state_dict