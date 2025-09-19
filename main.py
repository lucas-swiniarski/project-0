import torch
from tqdm import tqdm
from tokenizers import Tokenizer
from transformer.model import MyTransformer
from dataset.data_loader import DataLoader
from tokenizers.processors import TemplateProcessing


# Data parameters
batch_size = 32
context_size = 256
tokenizer_path = '/home/lucas/tokenizer/v1/tokenizer.json'
tokenized_data_dir = '/home/lucas/data/v1/tokenized/v2'

# Model hyperparameters
d_model = 512
num_attn_layers=12
num_query_heads=16
num_key_value_groups=4
expansion_factor = 4
dropout = 0.2
        
# Optimization hyperparametrs
max_iters = 55000
eval_interval = 500
eval_iter = 10
learning_rate = 3e-4

@torch.no_grad()
def estimate_loss(model, tokenizer, data_laoder, eval_interval, device):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iter)
        for k in range(eval_iter):
            X, Y = data_laoder.get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
        
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(f'Generation... :{tokenizer.decode(model.generate(context, max_new_tokens=500)[0].tolist())}')
    model.train()
    return out

def main():
    """
    Main function to instantiate and test the MyTransformer model.
    """
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
    ).to(device)
    print("Model instantiated successfully.")
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    
    print("Loading data...")
    data_loader = DataLoader(tokenized_data_dir, context_size, batch_size, device, device)
    n_batches_per_epoch = data_loader.get_n_batches_per_epoch()
    print(f"One epoch every {n_batches_per_epoch} batches.")

    # Create a dummy input tensor
    # Vocabulary size is 32000, so random integers are in the range [0, 31999]    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Wrap the training loop with tqdm for a progress bar
    with tqdm(range(max_iters), desc="Training", unit="step") as pbar:
        for step in pbar:
            if step % eval_interval == 0 or step == max_iters - 1:
                losses = estimate_loss(model, tokenizer, data_loader, eval_interval, device)
                print(f"\nstep {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            x, y = data_loader.get_batch('train')
            mask = torch.tril(torch.ones(context_size, context_size, device=device))
            logits, loss = model(x, y, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Update the progress bar description with the current loss
            pbar.set_postfix(loss=f"{loss.item():.4f}")


if __name__ == "__main__":
    main()