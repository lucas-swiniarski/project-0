import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Callable
from enum import Enum

from .model_utils import sample_next_token

class TrainingMode(Enum):
    SFT = 'sft'
    LORA = 'lora'
    EVAL = 'eval'

class MyTransformer(nn.Module):
    """
    A simple transformer.
    
    TODO(swiniarski): Implement:
    - Experts
    """

    def __init__(self, 
                 vocab_size: int = 32000, 
                 d_model: int = 256,
                 context_size: int = 512,
                 num_attn_layers: int = 12,
                 num_query_heads: int = 16,
                 num_key_value_groups: int = 4,
                 expansion_factor: int = 4,
                 dropout_rate: float = 0.0,
                 lora_rank: int = 0,
                 lora_alpha: float = 1.0,
                 ):
        super().__init__()
        self.context_size = context_size    
        self.token_embedding = TokenEmbeddingModel(vocab_size, d_model, context_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, num_query_heads, num_key_value_groups, expansion_factor, dropout_rate,
                lora_rank, lora_alpha
            ) for _ in range(num_attn_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.W_o = nn.Linear(d_model, vocab_size)
        self.mode = TrainingMode.SFT # Default to SFT

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, 
                idx: torch.Tensor, 
                targets: torch.Tensor | None = None, 
                mask: torch.Tensor | None = None,
                pos: torch.Tensor | None = None,
                keys_values: list[tuple[torch.Tensor]] | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transformer forward pass.
        
        Expected args in different configurations
        During training: idx is (B, T). Targets is (B, T). Mask is (T, T).
        
        During prefill: idx is (B, T). Masks is (T, T).
        
        During decode: idx is (B, 1). pos is (B, 1). keys_values is set.
        
        Args:
            idx (torch.Tensor): Token indices of size (B, T)
            targets (torch.Tensor, optional): Targets to compute loss of size (B, T). Defaults to None.
            mask (torch.Tensor, optional): Attention mask (e.g. causal) of size (T, T). Defaults to None (encoder-like).
            pos (torch.Tensor, optional): Position indices of size (B, T). Defaults to None.
            keys_values (list[tuple[torch.Tensor]], optional): pre-computed keys & values. Defaults to None.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: logits of size (B, T, vocab_size) and loss (scalar).
        """
        x = self.token_embedding(idx, pos=pos)
        new_keys_values = []
        for i, block in enumerate(self.blocks):
            x, key_value = block(x, 
                                 mask=mask, 
                                 key_value=keys_values[i] if keys_values is not None else None)
            new_keys_values += [key_value]
        x = self.ln(x)
        logits = self.W_o(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss, new_keys_values
        
    def _autoregressive_loop(self,
                             idx: torch.Tensor,
                             use_cache: bool,
                             step_handler: Callable) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Private helper for autoregressive operations (generation and scoring).

        Args:
            idx (torch.Tensor): The initial context, shape (B, T).
            use_cache (bool): If True, use KV caching for faster inference.
            step_handler (callable): A function that takes (logits, step_index)
                                     and returns (next_token, log_probability, should_continue).
        Returns:
            A tuple containing:
            - The final sequence (context + new tokens).
            - The log probabilities of the new tokens.
        """
        B, T = idx.shape
        log_probs = []
        keys_values = None
        decode_step = 0
        should_continue = True

        while should_continue:
            T = idx.shape[1]
            if T >= self.context_size:
                print(f"Sequence length ({T}) reached context size ({self.context_size}). Stopping generation.")
                break

            inference_mode = 'prefill' if not use_cache or keys_values is None else 'decode'
            pos = torch.full((B, 1), T - 1, dtype=torch.long, device=idx.device) if inference_mode == 'decode' else None
            mask = torch.tril(torch.ones(T, T, device=idx.device)) if inference_mode == 'prefill' else None
            keys_values = keys_values if inference_mode == 'decode' else None
            logits, _, keys_values = self.forward(
                idx[:, -1:] if inference_mode == 'decode' else idx,
                pos=pos,
                mask=mask, 
                keys_values=keys_values) # (B, T, vocab_size)
            
            next_token, log_prob, should_continue = step_handler(logits, decode_step)
            idx = torch.cat((idx, next_token), dim=1)
            log_probs.append(log_prob)
            decode_step += 1

        return idx, torch.cat(log_probs, dim=1) if log_probs else torch.empty((B, 0))
    
    def generate(self, 
                 idx: torch.Tensor, 
                 max_new_tokens: int, 
                 top_k: int = 0, 
                 top_p: float = 0.0, 
                 temperature: float = 1.0, 
                 use_cache: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates max_new_tokens from an input sentence idx.
        
        Args:
            idx (torch.Tensor): The initial context, shape (B, T).
            max_new_tokens (int): The maximum number of new tokens to generate.
            temperature (float, optional): Controls randomness. Higher values ( > 1.0) make output more random,
                                           lower values ( < 1.0) make it more deterministic. Defaults to 1.0.
            top_k (int, optional): If set, samples from the `k` most likely next tokens. Defaults to None.
            top_p (float, optional): If set, samples from the smallest set of tokens whose cumulative probability
                                     exceeds `p`. Defaults to None.
            use_cache (bool, optional): If set, use KV caching. Defaults to True.
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: The generated sequence, shape (B, T+max_new_tokens) and associated log-probs.
        """
        B, T_init = idx.shape
        if T_init >= self.context_size:
            print(f"Input sentences length ({T_init}) is too close to context size ({self.context_size}). Cannot generate tokens.")
            return idx, torch.empty((B, 0))
        
        n_to_gen = min(max_new_tokens, self.context_size - T_init)
        if n_to_gen < max_new_tokens:
            print(f"Warning: Asked to generate {max_new_tokens} tokens, but can only generate {n_to_gen} to stay within context size.")

        def sampling_handler(logits, decode_step):
            next_token, log_prob = sample_next_token(logits, top_k, top_p, temperature)
            return next_token, log_prob, decode_step < n_to_gen - 1

        return self._autoregressive_loop(idx, use_cache, sampling_handler)

    def score(self, 
              prompt: torch.Tensor, 
              completion: torch.Tensor, 
              use_cache: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        B, T_comp = completion.shape
        
        def scoring_handler(logits, decode_step):
            # Get the actual next token from the completion sequence
            target_token = completion[:, decode_step:decode_step+1]
            
            # Calculate the log probability of that target token
            # Note: temperature is not used in scoring, as we want the pure model probability
            probs = F.softmax(logits[:, -1, :], dim=-1) # Shape: (B, vocab_size)
            log_prob = torch.log(torch.gather(probs, 1, target_token))

            # Return the target token to "force" it as the next in the sequence
            return target_token, log_prob, decode_step < T_comp - 1

        return self._autoregressive_loop(prompt, use_cache, scoring_handler)
              
    def set_train_mode(self, mode: TrainingMode):
        """
        Sets the training mode for the model, freezing or unfreezing parameters accordingly.
        - SFT: All parameters are trainable.
        - LORA: Only LoRA-specific parameters are trainable (to be implemented).
        - EVAL: All parameters are frozen.
        
        All set model to .train() or .eval() for e.g. dropout or batchnorm.
        """
        self.train()
        self.mode = mode
        if mode == TrainingMode.SFT:
            print("Setting mode to SFT. All parameters are trainable.")
            for param in self.parameters():
                param.requires_grad = True
        elif mode == TrainingMode.EVAL:
            self.eval()
            print("Setting mode to EVAL. All parameters are frozen.")
            for param in self.parameters():
                param.requires_grad = False
        elif mode == TrainingMode.LORA:
            print("Setting mode to LORA. Freezing all non-LoRA parameters.")
            for param in self.parameters():
                param.requires_grad = False
            for name, param in self.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = True
            print(f"LoRA mode enabled. Trainable parameters: "
                  f"{sum(p.numel() for p in self.parameters() if p.requires_grad)/1e3:.2f}K")


class TokenEmbeddingModel(nn.Module):
    """
    A simple embedding model to represent tokens inside the transformer.
    
    TODO: Compare concat n_embd/2 pos+token emb to sum n_emb pos+token emb methods.
    """ 


    def __init__(self, 
                 vocab_size: int = 32000, 
                 n_embd: int = 256, 
                 context_size: int = 512):
        """init tokenization of transformer (token + position).

        Args:
            vocab_size (int, optional): # of unique tokens. Defaults to 32000.
            n_embd (int, optional): embedding size. Defaults to 256.
            context_size (int, optional): # of tokens in context of transformer. Defaults to 512.
        """
        super().__init__()
        self.context_size = context_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(context_size, n_embd)

    def forward(self, idx: torch.Tensor, pos: torch.Tensor | None = None):
        """Embed token indicies and positions.

        Args:
            idx (_type_): Tensor of indices, size (B, T).
            pos (_type_, optional): Tensor of positions to override default positions, size (B, T). Defaults to None.

        Returns:
            _type_: _description_
        """
        B, T = idx.shape
        assert T <= self.context_size, \
            f"Input sequence length ({T}) exceeds model's context size ({self.context_size})"
        device = idx.device
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        
        if pos is None:
            pos = torch.arange(T, device=device)
        pos_emb = self.position_embedding_table(pos) # (B, T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd)
        return x

class TransformerBlock(nn.Module):
    """
    A simple transformer block.
    """
    
    def __init__(self, 
                 d_model: int = 256, 
                 num_query_heads: int = 16, 
                 num_key_value_groups: int = 4, 
                 expansion_factor: int = 4,
                 dropout_rate: float = 0.0,
                 lora_rank: int = 0,
                 lora_alpha: float = 1.0,
                 ):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_model, num_query_heads, num_key_value_groups, dropout_rate, lora_rank, lora_alpha)
        self.ffn = FeedForward(d_model, expansion_factor, dropout_rate)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, key_value=None):
        # First sub-layer: Multi-Head Attention with residual connection
        attn_output, new_key_value = self.attn(self.ln1(x), mask=mask, key_value=key_value)
        x = x + attn_output
        # Second sub-layer: Feed-Forward Network with residual connection
        x = x + self.ffn(self.ln2(x))
        return x, new_key_value

class FeedForward(nn.Module):
    """
    A simple Feed Forward module.
    """
    def __init__(self, d_model: int = 256, expansion_factor: int = 4, dropout_rate: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.expansion_factor = expansion_factor
        self.linear_1 = nn.Linear(d_model, expansion_factor * d_model)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(expansion_factor * d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha):
        super().__init__()
        # todo: Read LoRA initialization values.
        self.lora_a = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_b = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = alpha / rank 

        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x):
        return (x @ self.lora_a @ self.lora_b) * self.scaling

class MultiHeadAttention(nn.Module):
    """
    A simple Multi-Head Attention module.
    """
    def __init__(self, 
                 d_model: int = 256, 
                 num_query_heads: int = 16, 
                 num_key_value_groups: int = 4, 
                 dropout_rate: float = 0.0,
                 lora_rank: int = 0,
                 lora_alpha: float = 1.0,
                 ):
        super().__init__()
        assert d_model % num_query_heads == 0, "d_model must be divisible by num_query_heads"
        assert num_query_heads % num_key_value_groups == 0, "num_query_heads must be divisible by num_key_value_groups"

        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_key_value_groups = num_key_value_groups
        
        self.head_dim = d_model // num_query_heads
        self.num_queries_per_group = self.num_query_heads // self.num_key_value_groups

        self.lora_rank = lora_rank

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, self.num_key_value_groups * self.head_dim)
        self.W_v = nn.Linear(d_model, self.num_key_value_groups * self.head_dim)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

        if self.lora_rank > 0:
            self.lora_q = LoRALayer(d_model, d_model, lora_rank, lora_alpha)
            self.lora_v = LoRALayer(d_model, self.num_key_value_groups * self.head_dim, lora_rank, lora_alpha)

    def forward(self, x, mask=None, key_value=None):
        B, T_q, C = x.shape

        # Original projections (B, T, C) -> (B, T, head_dim * n_heads**). 
        # n_heads** depends on groups, key & value vs query.
        q_proj = self.W_q(x)
        k_proj = self.W_k(x)
        v_proj = self.W_v(x)

        # Add LoRA if enabled
        if self.lora_rank > 0:
            q_proj = q_proj + self.lora_q(x)
            v_proj = v_proj + self.lora_v(x)
        
        T_kv = T_q
        new_key_value = (k_proj, v_proj)
        if key_value:
            # key_value = tuple[(B, T_cached, head_dim * n_head), (B, T_cached, head_dime * n_head)]
            cached_k_proj, cached_v_proj = key_value
            B_cached, T_cached, _ = cached_k_proj.shape
            assert B_cached == B, f'KV cache batch dim doesnt match input batch dim: input {B}, cache {B_cached}'
            k_proj = torch.cat((cached_k_proj, k_proj), dim=1)
            v_proj = torch.cat((cached_v_proj, v_proj), dim=1)
            T_kv = T_cached + T_q
            new_key_value = (k_proj, v_proj)

        # Project and reshape for multi-head attention
        Q = q_proj.view(B, T_q, self.num_query_heads, self.head_dim)
        K = k_proj.view(B, T_kv, self.num_key_value_groups, self.head_dim)
        V = v_proj.view(B, T_kv, self.num_key_value_groups, self.head_dim)

        # Repeat K and V for Grouped-Query Attention        
        if self.num_key_value_groups > 1:
            K = K.repeat_interleave(self.num_queries_per_group, dim=2)
            V = V.repeat_interleave(self.num_queries_per_group, dim=2)
            
        # Compute attention weights:
        Q = Q.transpose(1, 2) # B, num_query, T, head_dim
        K = K.transpose(1, 2) # B, num_query, T, head_dim 
        attn_probs = Q @ K.transpose(-1, -2) * self.head_dim**-0.5  # B, num_query, T, T
        if mask is not None:
            # Mask has shape (T_q, T_kv)
            attn_probs = attn_probs.masked_fill(mask[:T_q, :T_kv] == 0, float('-inf'))
        attn_probs = torch.softmax(attn_probs, dim=-1)
        attn_probs = self.dropout(attn_probs)
    
        # Concatenate heads and apply final linear layer
        V = V.transpose(1, 2) # B, num_query, T, head_dim
        attn_output = attn_probs @ V # B, num_query, T, head_Dim
        attn_output = attn_output.transpose(1, 2).reshape(B, T_q, self.d_model)
        output = self.dropout(self.W_o(attn_output))
        return output, new_key_value