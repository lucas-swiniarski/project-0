

pre_training_25_10_11_mode_config = {
    # Hyper-params
    'd_model': 384,
    'num_attn_layers': 6,
    'num_query_heads': 6,
    'num_key_value_groups': 3,
    'expansion_factor': 4,
    'dropout_rate': 0.2,
    # If lora_rank > 0, train a lora adapter (usually on a pre-trained model).
    'lora_rank': 0,
    'lora_alpha': 1.0,
    # Data parameters, to initialize embeddings.
    'vocab_size': 32000, # Will be set dynamically from the tokenizer
    'context_size': 512,
}