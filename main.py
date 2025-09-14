import torch
from model import MultiHeadAttention

def main():
    """
    Main function to instantiate and test the MultiHeadAttention model.
    """
    # Model parameters
    d_model = 512  # Dimension of the model
    num_heads = 8  # Number of attention heads
    batch_size = 10
    seq_length = 20

    print(f"Instantiating MultiHeadAttention with d_model={d_model} and num_heads={num_heads}")
    model = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    print("Model instantiated successfully.")

    # You can add more testing logic here, like creating dummy tensors and running a forward pass.

if __name__ == "__main__":
    main()