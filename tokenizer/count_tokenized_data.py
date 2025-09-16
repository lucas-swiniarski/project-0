import os
import argparse
from datasets import load_from_disk

def count_tokens_in_split(dataset_path):
    """
    Loads a tokenized dataset and counts the total number of tokens.
    """
    try:
        dataset = load_from_disk(dataset_path)
        total_tokens = sum(len(x) for x in dataset['input_ids'])
        return total_tokens
    except FileNotFoundError:
        print(f"Warning: Dataset not found at {dataset_path}. Skipping.")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Count the number of tokens in tokenized datasets.")
    parser.add_argument(
        '--tokenized-dir',
        type=str,
        default='/home/lucas/data/v1/tokenized/v2',
        help='Path to the directory containing the tokenized train, validation, and test sets.'
    )
    args = parser.parse_args()

    print(f"Counting tokens in datasets from: {args.tokenized_dir}\n")

    total_tokens_all_splits = 0
    for split in ["train", "validation", "test"]:
        split_path = os.path.join(args.tokenized_dir, split)
        num_tokens = count_tokens_in_split(split_path)
        if num_tokens > 0:
            print(f"  - {split:<12}: {num_tokens:,} tokens")
            total_tokens_all_splits += num_tokens

    print("\n" + "="*35)
    print(f"Total tokens across all splits: {total_tokens_all_splits:,}")

if __name__ == "__main__":
    main()