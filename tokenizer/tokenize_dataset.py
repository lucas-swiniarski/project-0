import argparse
import os

from datasets import load_from_disk
from tokenizers import Tokenizer

import tokenizer.profiles as tokenizer_profiles


def main():
    parser = argparse.ArgumentParser(description="Tokenize datasets using a trained tokenizer.")
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='/home/lucas/data/v2/raw/pre_training',
        help='Path to the directory containing dataset folders (e.g., institutional-books-1.0, tinystories).'
    )
    parser.add_argument(
        '--dataset-names',
        type=str,
        nargs='+',
        default=['all'],
        help='List of dataset names to tokenize, or "all" to tokenize all datasets in dataset-dir.'
    )
    parser.add_argument(
        '--tokenizer-path',
        type=str,
        default='/home/lucas/tokenizer/v1/tokenizer.json',
        help='Path to the trained tokenizer file.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/home/lucas/data/v2/tokenized',
        help='Base directory to save the tokenized datasets.'
    )
    parser.add_argument(
        '--num-proc',
        type=int,
        default=4,
        help='Number of processes to use for tokenization.'
    )
    parser.add_argument(
        '--tokenizer-profile',
        type=str,
        default='sentence_piece_v2',
        help='Name of the tokenizer profile to use.'
    )
    parser.add_argument(
        '--tokenizer-mode',
        type=str,
        default='pre_training',
        help='Mode used for tokenizing, depends on the tokenizer profile.'
    )
    args = parser.parse_args()

    # --- 1. Load tokenizer using profile ---
    print(f"Loading tokenizer from {args.tokenizer_path}, profile {args.tokenizer_profile} mode {args.tokenizer_mode}...")
    tokenizer_profile = tokenizer_profiles.TOKENIZER_NAME_TO_PROFILE[args.tokenizer_profile]()
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    tokenizer = tokenizer_profile.configure_tokenizer(tokenizer)

    print(f"Tokenizer vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Special tokens: {tokenizer_profile.get_special_tokens()}")
    print(f"Pad token: {tokenizer_profile.get_pad_token()}")
    print(f"Stop token: {tokenizer_profile.get_stop_token()}")

    # Determine which datasets to process
    if args.dataset_names == ['all']:
        # Get all subdirectories in dataset_dir
        dataset_names = [
            d for d in os.listdir(args.dataset_dir)
            if os.path.isdir(os.path.join(args.dataset_dir, d))
        ]
        print(f"Found {len(dataset_names)} datasets in {args.dataset_dir}: {dataset_names}")
    else:
        dataset_names = args.dataset_names
        print(f"Processing {len(dataset_names)} specified dataset(s): {dataset_names}")

    # Load all datasets from all specified dataset names
    raw_datasets = {}
    for dataset_name in dataset_names:
        dataset_path = os.path.join(args.dataset_dir, dataset_name)

        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset directory not found: {dataset_path}, skipping...")
            continue

        print(f"\nLoading dataset: {dataset_name}")
        for split in ["train", "validation", "test"]:
            split_path = os.path.join(dataset_path, split)
            if os.path.exists(split_path):
                print(f"  Loading {split} from {split_path}...")
                dataset_key = f"{dataset_name}/{split}"
                raw_datasets[dataset_key] = load_from_disk(split_path)
            else:
                print(f"  Warning: {split} split not found at {split_path}, skipping...")

    # Initialize args like this, otherwise args gets lost in parrallel processing .map not sure why.
    
    local_tokenizer = Tokenizer.from_file(args.tokenizer_path)
    local_profile = tokenizer_profiles.TOKENIZER_NAME_TO_PROFILE[args.tokenizer_profile]()
    local_tokenizer = local_profile.configure_tokenizer(local_tokenizer)

    def tokenize_function(examples):
        # Reload tokenizer and profile in each process to ensure correct state
        return local_profile.tokenize_datasets(
            examples, 
            tokenizer=local_tokenizer, 
            mode=args.tokenizer_mode
        )

    # --- 2. Tokenize datasets ---
    print("\nTokenizing datasets...")
    tokenized_datasets = {}
    token_counts = {}

    for dataset_key, dataset in raw_datasets.items():
        print(f"Tokenizing {dataset_key}...")
        tokenized_datasets[dataset_key] = dataset.map(
            tokenize_function,  # Use the new helper function
            batched=True,
            remove_columns=tokenizer_profile.tokenized_columns_to_remove(args.tokenizer_mode),
            num_proc=args.num_proc
        )

        # Count total tokens in this dataset/split
        total_tokens = sum(len(example) for example in tokenized_datasets[dataset_key]['input_ids'])
        num_examples = len(tokenized_datasets[dataset_key])
        avg_tokens = total_tokens / num_examples if num_examples > 0 else 0

        token_counts[dataset_key] = {
            'total_tokens': total_tokens,
            'num_examples': num_examples,
            'avg_tokens_per_example': avg_tokens
        }

        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Number of examples: {num_examples:,}")
        print(f"  Average tokens per example: {avg_tokens:.2f}")

        # --- 3. Save tokenized datasets ---
        # Save to: output_dir/dataset-name/split
        output_path = os.path.join(args.output_dir, dataset_key)
        os.makedirs(output_path, exist_ok=True)
        print(f"Saving tokenized {dataset_key} to {output_path}...")
        tokenized_datasets[dataset_key].save_to_disk(output_path)

    print("\n" + "="*60)
    print("TOKENIZATION COMPLETE")
    print("="*60)

    # Print summary statistics
    print("\nToken count summary:")
    grand_total_tokens = 0
    grand_total_examples = 0

    for dataset_key, counts in token_counts.items():
        print(f"\n{dataset_key}:")
        print(f"  Total tokens: {counts['total_tokens']:,}")
        print(f"  Number of examples: {counts['num_examples']:,}")
        print(f"  Average tokens per example: {counts['avg_tokens_per_example']:.2f}")
        grand_total_tokens += counts['total_tokens']
        grand_total_examples += counts['num_examples']

    print(f"\n{'-'*60}")
    print(f"GRAND TOTAL:")
    print(f"  Total tokens across all datasets: {grand_total_tokens:,}")
    print(f"  Total examples across all datasets: {grand_total_examples:,}")
    if grand_total_examples > 0:
        print(f"  Overall average tokens per example: {grand_total_tokens/grand_total_examples:.2f}")

if __name__ == "__main__":
    main()
