import torch
from tqdm import tqdm
from torch.amp.grad_scaler import GradScaler
import tokenizer.profiles as tokenizer_profiles
from transformer.model import MyTransformer, TrainingMode
from transformer import model_utils
from tokenizers import Tokenizer
from dataset.post_training.sft.data_loader import DataLoader
import os
import versioned_model_configs

# Path to a pre-trained model for SFT. This is required.
base_model_path = '/home/lucas/project-0/checkpoints/25_10_11_model/model_step_24999.pt'
model_config = versioned_model_configs.pre_training_25_10_11_mode_config # Configuration of base_model_path.

# SFT specific configuration overrides
sft_config = {
    'dropout_rate': 0.2,
    # If lora_rank > 0, train a lora adapter (usually on a pre-trained model).
    'lora_rank': 0,
    'lora_alpha': 1.0,
    'context_size': model_config['context_size'],
}

gen_params = {
    'max_new_tokens': 128,
    'top_k': 64,
    'top_p': 0.9,
    'temperature': 1.0
}

# Data parameters
batch_size = 16 # SFT batches with padding can take more memory
context_size = model_config['context_size']
tokenizer_path = '/home/lucas/tokenizer/v1/tokenizer.json'
tokenizer_profile_name = 'post_training_v1'
tokenized_data_dir = '/home/lucas/data/v1/tokenized/post_training/sft'

# Optimization hyperparametrs
max_iters = 6000
eval_interval = 50
eval_batches = 50
learning_rate = 1e-4 # Usually lower for fine-tuning

# Checkpointing
checkpoint_dir = '/home/lucas/project-0/checkpoints/25_10_11_pt_sft1/'

def main():
    """
    Main function to run Supervised Fine-Tuning (SFT) on the MyTransformer model.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer_profile = tokenizer_profiles.TOKENIZER_NAME_TO_PROFILE[tokenizer_profile_name]()
    tokenizer = tokenizer_profile.configure_tokenizer(tokenizer)
    new_vocab_size = tokenizer.get_vocab_size()
    pad_token_id = tokenizer.token_to_id("[PAD]")
    assert pad_token_id is not None, "[PAD] token not found in tokenizer"
    
    # Set a seed for reproducibility
    torch.manual_seed(0)

    # --- Model Loading and Configuration ---
    print(f"Loading pre-trained model from {base_model_path}")
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Base model not found at {base_model_path}. SFT requires a pre-trained model.")

    state_dict = torch.load(base_model_path, map_location=device)
    model_config = versioned_model_configs.pre_training_25_10_11_mode_config
    # To change once new models trained
    # checkpoint = torch.load(base_model_path, map_location=device)
    # model_config = checkpoint['model_config']
    # state_dict = checkpoint['model_state_dict']
        
    # --- Handle Vocabulary Expansion ---
    old_vocab_size = model_config['vocab_size']
    state_dict = model_utils.expand_model_vocabulary(state_dict, old_vocab_size, new_vocab_size)

    # Update config with new vocab size and SFT-specific overrides
    model_config['vocab_size'] = new_vocab_size
    model_config.update(sft_config)

    print("Initializing transformer with loaded configuration...")
    model = MyTransformer(**model_config).to(device)
    
    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    if incompatible_keys.missing_keys:
        print("Warning: The following keys were not found in the checkpoint and will be randomly initialized:")
        print(f"  {incompatible_keys.missing_keys}")
    if incompatible_keys.unexpected_keys:
        print("Warning: The following keys were in the checkpoint but not in the model:")
        print(f"  {incompatible_keys.unexpected_keys}")

    print("Model instantiated successfully.")
    print(f"Vocabulary size: {model_config['vocab_size']}")
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M total parameters.")
    
    # --- Set Training Mode and Optimizer ---
    train_mode = TrainingMode.LORA if model_config['lora_rank'] > 0 else TrainingMode.SFT
    model.set_train_mode(train_mode)

    # Create optimizer for trainable parameters only
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    print("Loading data...")
    data_loader = DataLoader(tokenized_data_dir, context_size, batch_size, pad_token_id, device)
    n_batches_per_epoch = data_loader.get_n_batches_per_epoch()
    print(f"One epoch every {n_batches_per_epoch} batches.")
    
    scaler = GradScaler(enabled=(device == 'cuda'))
    
    # Create the causal attention mask once outside the loop
    # The SFT dataloader uses padding, so we don't need a static mask here.
    # The model's forward pass should handle creating the mask from padding tokens.
    # For now, we pass None and rely on the dataloader's truncation.
    # A better approach would be to handle padding masks in the model.
    
    with tqdm(range(max_iters), desc="Training", unit="step", ncols=120) as pbar:
        for step in pbar:
            if step > 0 and (step % eval_interval == 0 or step == max_iters - 1):
                model_utils.generate_text(model, tokenizer, data_loader, context_size, **gen_params)
                losses = model_utils.estimate_loss(model, data_loader, eval_batches)
                print(f"\nstep {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                
                checkpoint_path = os.path.join(checkpoint_dir, f'model_step_{step}.pt')
                print(f"Saving checkpoint to {checkpoint_path}")
                checkpoint = {
                    'model_config': model_config,
                    'model_state_dict': model.state_dict(),
                }
                torch.save(checkpoint, checkpoint_path)
            
            x, y = data_loader.get_batch('train')
            
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == 'cuda')):
                logits, loss, _ = model(x, y) # Mask is not needed as SFT data is padded to context_size

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

if __name__ == "__main__":
    main()
