import torch
from tqdm import tqdm
from torch.amp.grad_scaler import GradScaler
import tokenizer.profiles as tokenizer_profiles
from transformer.model import MyTransformer, TrainingMode
from transformer import model_utils
from tokenizers import Tokenizer
from dataset.pre_training.data_loader import DataLoader
import os

model_config = {
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
    'vocab_size': -1, # Will be set dynamically from the tokenizer
    'context_size': 512,
}

gen_params = {
    'max_new_tokens': 128,
    'top_k': 64,
    'top_p': 0.9,
    'temperature': 1.0
}

# Data parameters
batch_size = 32
context_size = model_config['context_size']
tokenizer_path = '/home/lucas/tokenizer/v1/tokenizer.json'
tokenizer_profile_name = 'pre_training_v1' # Or 'post_training_v1'
tokenized_data_dir = '/home/lucas/data/data/v1/tokenized/pre_training'

# Optimization hyperparametrs
max_iters = 25000
eval_interval = 500
eval_batches = 50
learning_rate = 3e-4

# Checkpointing
checkpoint_dir = './checkpoints/25_10_11_model/'
base_model_path = '' # './checkpoints/25_09_23_model/model_step_46000.pt' Path to a pre-trained model for continue fine-tuning or LoRA.

def main():
    """
    Main function to instantiate and test the MyTransformer model.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer_profile = tokenizer_profiles.TOKENIZER_NAME_TO_PROFILE[tokenizer_profile_name]()
    tokenizer = tokenizer_profile.configure_tokenizer(tokenizer)
    model_config['vocab_size'] = tokenizer.get_vocab_size()

    print(f"Vocabulary size: {model_config['vocab_size']}")
    # Set a seed for reproducibility
    torch.manual_seed(0)

    print(f"Init transformer...")
    model = MyTransformer(**model_config).to(device)

    # Load a base model if specified (e.g., for LoRA training)
    if base_model_path:
        print(f"Loading pre-trained model from {base_model_path}")
        state_dict = torch.load(base_model_path, weights_only=True)
        incompatible_keys = model.load_state_dict(state_dict, strict=False)
        if incompatible_keys.missing_keys:
            # Here we'd expect this warning to trigger only when starting a LoRA training.
            # Reason is:
            # - When doing sft, LoRA weights aren't initialized.
            # - When starting LoRA from sft ckpt, LoRA weights aren't initialized thus trigger warning.
            # - When continuing both sft and LoRA, the weights should already all be initalized.
            print("Warning: The following keys were not found in the checkpoint and will be randomly initialized:")
            print(f"  {incompatible_keys.missing_keys}")

    print("Model instantiated successfully.")
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M total parameters.")
    
    # --- Set Training Mode and Optimizer ---
    train_mode = TrainingMode.LORA if model_config['lora_rank'] > 0 else TrainingMode.SFT
    model.set_train_mode(train_mode)

    # Create optimizer for trainable parameters only
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    print("Loading data...")
    data_loader = DataLoader(tokenized_data_dir, context_size, batch_size, device)
    n_batches_per_epoch = data_loader.get_n_batches_per_epoch()
    print(f"One epoch every {n_batches_per_epoch} batches.")

    # Create a dummy input tensor
    # Vocabulary size is 32000, so random integers are in the range [0, 31999]    
    scaler = GradScaler(device, enabled=(device == 'cuda'))
    
    # Create the causal attention mask once outside the loop
    mask = torch.tril(torch.ones(context_size, context_size, device=device))
    # Wrap the training loop with tqdm for a progress bar
    
    with tqdm(range(max_iters), desc="Training", unit="step", ncols=120) as pbar:
        for step in pbar:
            if step % eval_interval == 0 or step == max_iters - 1:
                generated_text = model_utils.generate_text(model, tokenizer, data_loader, context_size, **gen_params)
                losses = model_utils.estimate_loss(model, data_loader, eval_batches)
                print(f"\nstep {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                
                
                checkpoint_path = os.path.join(checkpoint_dir, f'model_step_{step}.pt')
                print(f"\nSaving checkpoint to {checkpoint_path}")
                checkpoint = {
                    'model_config': model_config,
                    'model_state_dict': model.state_dict(),
                }
                torch.save(checkpoint, checkpoint_path)
            
            x, y = data_loader.get_batch('train')
            
            # Mixed precision training context
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == 'cuda')):
                logits, loss, _ = model(x, y, mask)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update the progress bar description with the current loss
            pbar.set_postfix(loss=f"{loss.item():.4f}")


if __name__ == "__main__":
    main()