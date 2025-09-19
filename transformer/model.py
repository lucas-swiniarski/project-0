import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
                 ):
        super().__init__()
        self.context_size = context_size    
        self.token_embedding = TokenEmbeddingModel(vocab_size, d_model, context_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, num_query_heads, num_key_value_groups, expansion_factor, dropout_rate) for _ in range(num_attn_layers)
        ])
        self.W_o = nn.Linear(d_model, vocab_size)
        
    def forward(self, idx, targets=None, mask=None):
        x = self.token_embedding(idx)
        for block in self.blocks:
            x = block(x, mask)
        logits = self.W_o(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        if max_new_tokens > self.context_size:
            print(f'Max context size of {self.context_size} lower than number of tokens asked {max_new_tokens}, trimming.')
        for _ in range(min(self.context_size, max_new_tokens)):
            logits, _ = self.forward(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
        

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

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.context_size, \
            f"Input sequence length ({T}) exceeds model's context size ({self.context_size})"
        device = idx.device
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embd)
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
                 dropout_rate: float = 0.0):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_query_heads, num_key_value_groups, dropout_rate)
        self.ffn = FeedForward(d_model, expansion_factor, dropout_rate)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x


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

class MultiHeadAttention(nn.Module):
    """
    A simple Multi-Head Attention module.
    """
    def __init__(self, d_model: int = 256, num_query_heads: int = 16, num_key_value_groups: int = 4, dropout_rate: float = 0.0):
        super().__init__()
        assert d_model % num_query_heads == 0, "d_model must be divisible by num_query_heads"
        assert num_query_heads % num_key_value_groups == 0, "num_query_heads must be divisible by num_key_value_groups"

        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_key_value_groups = num_key_value_groups
        
        self.head_dim = d_model // num_query_heads
        self.num_queries_per_group = self.num_query_heads // self.num_key_value_groups

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, self.num_key_value_groups * self.head_dim)
        self.W_v = nn.Linear(d_model, self.num_key_value_groups * self.head_dim)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        B, T, C = x.shape

        # Project and reshape for multi-head attention
        Q = self.W_q(x).view(B, T, self.num_query_heads, self.head_dim)
        K = self.W_k(x).view(B, T, self.num_key_value_groups, self.head_dim)
        V = self.W_v(x).view(B, T, self.num_key_value_groups, self.head_dim)

        # Repeat K and V for Grouped-Query Attention        
        if self.num_key_value_groups > 1:
            K = K.repeat_interleave(self.num_queries_per_group, dim=2)
            V = V.repeat_interleave(self.num_queries_per_group, dim=2)
            
        # Compute attention weights:
        Q = Q.transpose(1, 2) # B, num_query, T, head_dim
        K = K.transpose(1, 2) # B, num_query, T, head_dim 
        attn_probs = Q @ K.transpose(-1, -2) * self.head_dim**-0.5  # B, num_query, T, T
        if mask is not None:
            attn_probs = attn_probs.masked_fill(mask == 0,float('-inf'))
        attn_probs = torch.softmax(attn_probs, dim=-1)
        attn_probs = self.dropout(attn_probs)
    
        # Concatenate heads and apply final linear layer
        V = V.transpose(1, 2) # B, num_query, T, head_dim
        attn_output = attn_probs @ V # B, num_query, T, head_Dim
        attn_output = attn_output.transpose(1, 2).reshape(B, T, self.d_model)
        output = self.dropout(self.W_o(attn_output))
        return output