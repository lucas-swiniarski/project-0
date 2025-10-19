import copy
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.amp.grad_scaler import GradScaler
import tokenizer.profiles as tokenizer_profiles
from transformer.model import MyTransformer, TrainingMode
import math
from transformer import model_utils
from transformer.tensorboard_utils import TensorBoardLogger
from tokenizers import Tokenizer
from dataset.post_training.rl.data_loader import DataLoader
import os
import versioned_model_configs

# Path to a pre-trained model for RL. This is required.
base_model_path = '/home/lucas/project-0/checkpoints/25_10_15_pt_sft1/model_step_5999.pt'
model_config = versioned_model_configs.post_training_25_10_15_mode_config # Configuration of base_model_path.

# RL specific configuration overrides
rl_config = {
    'dropout_rate': 0.2,
    # If lora_rank > 0, train a lora adapter (usually on a pre-trained model).
    'lora_rank': 0,
    'lora_alpha': 1.0,
    'context_size': model_config['context_size'],
}

gen_params = {
    'max_new_tokens': 256,
    'top_k': 64,
    'top_p': 0.9,
    'temperature': 1.0
}

# Loss params
beta_dpo = 1.0

# Data parameters
batch_size = 2 # RL batches with padding can take more memory
context_size = model_config['context_size']
tokenizer_path = '/home/lucas/tokenizer/v1/tokenizer.json'
tokenizer_profile_name = 'post_training_v1'
tokenized_data_dir = '/home/lucas/data/v1/tokenized/post_training/rl'

# Iterations params
max_iters = 6000
eval_interval = 500
eval_batches = 20 # Reduced to speed up eval loop

# Optimizer params.
learning_rate = 3e-4 # Max learning rate
warmup_iters = 500
lr_decay_iters = max_iters # Should be >= max_iters
min_lr = 1e-6 # Final learning rate
grad_clip = 1.0 # Clip gradients at this value

# Checkpointing
checkpoint_dir = '/home/lucas/project-0/checkpoints/25_10_15_pt_sft1/'

def get_lr(it: int) -> float:
    """Calculates the learning rate for a given iteration."""
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (learning_rate - min_lr)

def main():
    """
    Main function to run DPO on MyTransformer model.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    tensorboard_log_dir = os.path.join(checkpoint_dir, 'tensorboard')
    logger = TensorBoardLogger(log_dir=tensorboard_log_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer_profile = tokenizer_profiles.TOKENIZER_NAME_TO_PROFILE[tokenizer_profile_name]()
    tokenizer = tokenizer_profile.configure_tokenizer(tokenizer)
    new_vocab_size = tokenizer.get_vocab_size()
    pad_token_id = tokenizer.token_to_id(tokenizer_profile.get_pad_token())
    stop_token_id = tokenizer.token_to_id(tokenizer_profile.get_stop_token())
    assert pad_token_id is not None, "pad token not found in tokenizer"
    assert stop_token_id is not None, "stop token not found in tokenizer"
    
    # Set a seed for reproducibility
    torch.manual_seed(1)

    # --- Model Loading and Configuration ---
    print(f"Loading model from {base_model_path}")
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Base model not found at {base_model_path}. RL requires a pre-trained model.")

    state_dict = torch.load(base_model_path, map_location=device)
    model_config = versioned_model_configs.pre_training_25_10_11_mode_config
    # To change once new models trained
    # checkpoint = torch.load(base_model_path, map_location=device)
    # model_config = checkpoint['model_config']
    # state_dict = checkpoint['model_state_dict']
        
    # --- Handle Vocabulary Expansion ---
    old_vocab_size = model_config['vocab_size']
    state_dict = model_utils.expand_model_vocabulary(state_dict, old_vocab_size, new_vocab_size)

    # Update config with new vocab size and RL-specific overrides
    model_config['vocab_size'] = new_vocab_size
    model_config.update(rl_config)

    print("Initializing transformer with loaded configuration...")
    pi_theta = MyTransformer(**model_config).to(device)
    
    incompatible_keys = pi_theta.load_state_dict(state_dict, strict=False)
    if incompatible_keys.missing_keys:
        print("Warning: The following keys were not found in the checkpoint and will be randomly initialized:")
        print(f"  {incompatible_keys.missing_keys}")
    if incompatible_keys.unexpected_keys:
        print("Warning: The following keys were in the checkpoint but not in the model:")
        print(f"  {incompatible_keys.unexpected_keys}")

    print("Model instantiated successfully.")
    print(f"Vocabulary size: {model_config['vocab_size']}")
    print(f"{sum(p.numel() for p in pi_theta.parameters())/1e6:.2f}M total parameters.")
    
    print("Copying pi_theta for pi_ref.")
    pi_ref = copy.deepcopy(pi_theta)
    
    # --- Set Training Mode and Optimizer ---
    train_mode = TrainingMode.LORA if model_config['lora_rank'] > 0 else TrainingMode.TRAIN
    pi_theta.set_train_mode(train_mode)
    pi_ref.set_train_mode(TrainingMode.EVAL)
   
    # Create optimizer for trainable parameters only
    trainable_params = [p for p in pi_theta.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    print("Loading data...")
    data_loader = DataLoader(tokenized_data_dir, context_size, batch_size, pad_token_id, device)
    n_batches_per_epoch = data_loader.get_n_batches_per_epoch()
    print(f"One epoch every {n_batches_per_epoch} batches.")
    
    scaler = GradScaler(enabled=(device == 'cuda'))
    
    # Create the causal attention mask once outside the loop
    mask = torch.tril(torch.ones(context_size, context_size, device=device))
    
    with tqdm(range(max_iters), desc="Training", unit="step", ncols=120) as pbar:
        for step in pbar:
            # determine and set the learning rate for this iteration
            lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            if step > 0 and (step % eval_interval == 0 or step == max_iters - 1):
                generated_text = model_utils.generate_text(
                    model, tokenizer, data_loader, context_size, stop_token=stop_token_id, **gen_params)
                logger.log_text('Generations/sample', generated_text, step)
                losses = model_utils.estimate_dpo_loss(model, data_loader, eval_batches)
                print(f"\nstep {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                logger.log_scalars('Loss/eval', {'train': losses['train'], 'val': losses['val']}, step)
                
                checkpoint_path = os.path.join(checkpoint_dir, f'model_step_{step}.pt')
                print(f"Saving checkpoint to {checkpoint_path}")
                checkpoint = {
                    'model_config': model_config,
                    'model_state_dict': pi_theta.state_dict(),
                }
                torch.save(checkpoint, checkpoint_path)
            
            (x_w, y_w), (x_l, y_l) = data_loader.get_batch('train')                        
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == 'cuda')):
                _, nll_theta_w, _ = pi_theta(x_w, y_w, mask, reduction='sum')
                _, nll_theta_l, _ = pi_theta(x_l, y_l, mask, reduction='sum')
                _, nll_ref_w, _ = pi_ref(x_w, y_w, mask, reduction='sum')
                _, nll_ref_l, _ = pi_ref(x_l, y_l, mask, reduction='sum')
                
                loss = - torch.log(
                    F.sigmoid(
                        beta_dpo * ((nll_ref_w - nll_theta_w) - (nll_ref_l - nll_theta_l))
                    )
                )


            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            # Unscale gradients before clipping to get the correct norm
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(pi_theta.parameters(), grad_clip)
            
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            # Log metrics to TensorBoard
            logger.log_scalar('Loss/train', loss.item(), step)
            logger.log_scalar('LearningRate', optimizer.param_groups[0]['lr'], step)
            logger.log_scalar('Gradients/norm', grad_norm.item(), step)
            logger.log_scalars('NLL', {
                'theta_w': nll_theta_w.item(),
                'ref_w': nll_ref_w.item(),
                'theta_l': nll_theta_l.item(),
                'ref_l': nll_ref_l.item(),
            }, step)

    logger.close()

if __name__ == "__main__":
    main()
