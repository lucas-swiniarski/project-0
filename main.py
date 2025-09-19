import torch
from transformer.model import MyTransformer

def main():
    """
    Main function to instantiate and test the MyTransformer model.
    """

    # Set device to GPU if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Set a seed for reproducibility
    torch.manual_seed(0)

    print(f"Init transformer...")
    model = MyTransformer().to(device) # Instantiate and move the model to the selected device
    print("Model instantiated successfully.")

    # Create a dummy input tensor
    # Batch size = 4, Context size = 512
    # Vocabulary size is 32000, so random integers are in the range [0, 31999]
    B, T = 4, 512
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, device=device)
    
    for step in range(4000):
        dummy_input = torch.randint(0, 32000, (B, T + 1), device=device)
        x, y = dummy_input[:, :-1], dummy_input[:, 1:]
        mask = torch.tril(torch.ones(T, T, device=device))
        logits, loss = model(x, y, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(f"Step {step + 1}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()