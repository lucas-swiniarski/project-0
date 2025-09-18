import torch
from transformer.model import MyTransformer

def main():
    """
    Main function to instantiate and test the MyTransformer model.
    """

    print(f"Init transformer...")
    model = MyTransformer()
    print("Model instantiated successfully.")

    # Create a dummy input tensor
    # Batch size = 4, Context size = 512
    # Vocabulary size is 32000, so random integers are in the range [0, 31999]
    B, T = 4, 512
    dummy_input = torch.randint(0, 32000, (B, T))
    mask = torch.tril(torch.ones(T, T))

    print(f"\nCreated a dummy input tensor of size: {dummy_input.shape}")
    print("Running a forward pass...")
    output = model(dummy_input)
    print(f"Forward pass successful. Output tensor shape: {output.shape}")

if __name__ == "__main__":
    main()