import torch
from tqdm import tqdm
from torch.amp.grad_scaler import GradScaler
from tokenizers import Tokenizer
from transformer.model import MyTransformer, TrainingMode
from dataset.data_loader import DataLoader
from tokenizers.processors import TemplateProcessing
import os


# Data parameters
batch_size = 64
context_size = 256
tokenizer_path = '/home/lucas/tokenizer/v1/tokenizer.json'
tokenized_data_dir = '/home/lucas/data/v1/tokenized/v2'

# Model hyperparameters
d_model = 384
num_attn_layers = 6
num_query_heads = 6
num_key_value_groups = 3
expansion_factor = 4
dropout = 0.2

# LoRA parameters
lora_rank = 0 # Set to 0 to disable LoRA
lora_alpha = 1.0

# Optimization hyperparametrs
max_iters = 55000
eval_interval = 500
eval_batches = 8
learning_rate = 3e-4

# Checkpointing
checkpoint_dir = './checkpoints/25_09_23_model/'
base_model_path = None # './checkpoints/25_09_22_model/model_step_500.pt' # Path to a pre-trained model for LoRA or fine-tuning

@torch.no_grad()
def estimate_loss(model, tokenizer, data_laoder, eval_batches, device):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_batches)
        for k in range(eval_batches):
            X, Y = data_laoder.get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
        
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(f'Generation... :{tokenizer.decode(model.generate(context, max_new_tokens=128)[0].tolist(), skip_special_tokens=False)}')
    model.train()
    return out

def main():
    """
    Main function to instantiate and test the MyTransformer model.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.post_processor = TemplateProcessing(
        single="$A", # The main sequence. We can add special tokens to the template if we want to see them.
        special_tokens=[
            ("[SOS]", tokenizer.token_to_id("[SOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )
    vocab_size = tokenizer.get_vocab_size()

    # Set a seed for reproducibility
    torch.manual_seed(0)

    print(f"Init transformer...")
    model = MyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        context_size=context_size,
        num_attn_layers=num_attn_layers,
        num_query_heads=num_query_heads,
        num_key_value_groups=num_key_value_groups,
        expansion_factor=expansion_factor,
        dropout_rate=dropout,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    ).to(device)

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
    train_mode = TrainingMode.LORA if lora_rank > 0 else TrainingMode.SFT
    model.set_train_mode(train_mode)

    # Create optimizer for trainable parameters only
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    print("Loading data...")
    data_loader = DataLoader(tokenized_data_dir, context_size, batch_size, device, device)
    n_batches_per_epoch = data_loader.get_n_batches_per_epoch()
    print(f"One epoch every {n_batches_per_epoch} batches.")

    # Create a dummy input tensor
    # Vocabulary size is 32000, so random integers are in the range [0, 31999]    
    scaler = GradScaler(device, enabled=(device == 'cuda'))
    
    # Create the causal attention mask once outside the loop
    mask = torch.tril(torch.ones(context_size, context_size, device=device))
    # Wrap the training loop with tqdm for a progress bar
    with tqdm(range(max_iters), desc="Training", unit="step") as pbar:
        for step in pbar:
            if step % eval_interval == 0 or step == max_iters - 1:
                # Save a checkpoint
                checkpoint_path = os.path.join(checkpoint_dir, f'model_step_{step}.pt')
                print(f"\nSaving checkpoint to {checkpoint_path}")
                torch.save(model.state_dict(), checkpoint_path)

                losses = estimate_loss(model, tokenizer, data_loader, eval_batches, device)
                print(f"\nstep {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            x, y = data_loader.get_batch('train')
            
            # Mixed precision training context
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == 'cuda')):
                logits, loss = model(x, y, mask)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update the progress bar description with the current loss
            pbar.set_postfix(loss=f"{loss.item():.4f}")


if __name__ == "__main__":
    main()