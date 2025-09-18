import torch
import torch.nn as nn
import math

class MyTransformer(nn.Module):
    """
    A simple transformer.
    
    TODO(swiniarski): Implement:
    - Experts
    - Chain attention + experts layers.
    """

    def __init__(self, 
                 vocab_size: int = 32000, 
                 d_model: int = 256,
                 context_size: int = 512,
                 num_query_heads: int = 16,
                 num_key_value_groups: int = 4,
                 ):
        super().__init__()
        
        self.token_embedding = TokenEmbeddingModel(vocab_size, d_model, context_size)
        self.one_attention_head = MultiHeadAttention(d_model, num_query_heads, num_key_value_groups)
        
    def forward(self, idx, mask=None, target=None):
        x = self.token_embedding(idx)
        x = self.one_attention_head(x, mask)
        return x
        

class TokenEmbeddingModel(nn.Module):
    """
    A simple embedding model to represent tokens inside the transformer.
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
        assert n_embd % 2 == 0, "n_embd must be even"
        # TODO: We likely dont want position to be as big as token embedding.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd // 2)
        self.position_embedding_table = nn.Embedding(context_size, n_embd // 2)

    def forward(self, idx):
        B, T = idx.shape
        device = idx.device

        tok_emb = self.token_embedding_table(idx) # B, T, C
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # T, C
        pos_emb = pos_emb.unsqueeze(0).repeat(B, 1, 1) # B, T, C
        x = torch.concat([tok_emb, pos_emb], axis=-1) # B, T, C * 2
        print(tok_emb.shape, pos_emb.shape, x.shape)
        return x


class MultiHeadAttention(nn.Module):
    """
    A simple Multi-Head Attention module.
    """
    def __init__(self, d_model: int = 256, num_query_heads: int = 16, num_key_value_groups: int = 4):
        super().__init__()
        assert d_model % num_query_heads == 0, "d_model must be divisible by num_query_heads"
        assert num_query_heads % num_key_value_groups == 0, "num_query_heads must be divisible by num_key_value_groups"

        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_key_value_groups = num_key_value_groups
        
        self.head_dim = d_model // num_query_heads
        self.num_queries_per_group = self.num_query_heads // self.num_key_value_groups

        self.W_q = nn.Linear(d_model, d_model)
        # The output dimension for K and V should match the number of KV heads times the head dimension
        self.W_k = nn.Linear(d_model, self.num_key_value_groups * self.head_dim)
        self.W_v = nn.Linear(d_model, self.num_key_value_groups * self.head_dim)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Note: In a real implementation, head_dim would be self.head_dim
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

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
        
        # Compute attention:
        # TODO: Continue from there
        attn_scores = Q @ K.transpose(-2, -1) 
        # (batch_size, num_query_heads, seq_len, head_dim)
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads and apply final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.W_o(attention_output)
        return output