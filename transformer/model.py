import torch
import torch.nn as nn
import math

class TokenEmbeddingModel(nn.Module):
    """
    A simple embedding model to represent tokens inside the transformer.
    """ 


    def __init__(self, vocab_size: int = 32000, n_embd: int = 256, block_size: int = 512):
        super.__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding()

    def forward(self, idx):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # B, T, C
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # T, C
        x = tok_emb + pos_emb # B, T, C
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

        self.d_k = d_model // num_query_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model // num_key_value_groups)
        self.W_v = nn.Linear(d_model, d_model // num_key_value_groups)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def forward(self, Q, K, V, mask=None):
        # This is a simplified forward pass. A full implementation would involve splitting heads.
        # For simplicity, we'll apply linear layers and then the attention mechanism.
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(attention_output)
        return output