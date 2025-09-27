import os
import argparse
from datasets import load_from_disk
from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer
from .utils import create_tokenizer

def get_training_corpus(dataset):
    """
    A generator function to yield text from the dataset, which is memory-efficient.
    """
    for i in range(0, len(dataset), 1000):
        yield dataset[i: i + 1000]["text"]

def main():
    parser = argparse.ArgumentParser(description="Train a new BPE tokenizer from scratch.")
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='/home/lucas/data/v1/train',
        help='Path to the training dataset directory.'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=32000,
        help='The desired vocabulary size for the tokenizer.'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='/home/lucas/tokenizer/v1/tokenizer.json',
        help='Path where the trained tokenizer will be saved.'
    )
    args = parser.parse_args()

    # --- 1. Initialize a new tokenizer ---
    tokenizer = create_tokenizer()

    # --- 2. Train the tokenizer ---
    print(f"Loading training dataset from {args.dataset_dir}...")
    train_dataset = load_from_disk(args.dataset_dir)

    trainer = BpeTrainer(vocab_size=args.vocab_size, special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"])
    
    print(f"Training tokenizer with vocab size {args.vocab_size}...")
    tokenizer.train_from_iterator(get_training_corpus(train_dataset), trainer=trainer)
    print("Training complete.")

    # --- 3. Save the tokenizer ---
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    tokenizer.save(args.output_path)
    print(f"Tokenizer saved to {args.output_path}")

if __name__ == "__main__":
    main()