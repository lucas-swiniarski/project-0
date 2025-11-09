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
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Batch size for processing and writing (default: 1000).'
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

    # Create tokenization function
    def tokenize_function(examples):
        return tokenizer_profile.tokenize_datasets(
            examples,
            tokenizer=tokenizer,
            mode=args.tokenizer_mode
        )

    # Process each dataset one at a time to avoid OOM
    print("\nTokenizing datasets...")
    token_counts = {}

    for dataset_name in dataset_names:
        dataset_path = os.path.join(args.dataset_dir, dataset_name)

        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset directory not found: {dataset_path}, skipping...")
            continue

        print(f"\nProcessing dataset: {dataset_name}")

        for split in ["train", "validation", "test"]:
            split_path = os.path.join(dataset_path, split)

            if not os.path.exists(split_path):
                print(f"  Warning: {split} split not found at {split_path}, skipping...")
                continue

            dataset_key = f"{dataset_name}/{split}"
            print(f"\n  Tokenizing {dataset_key}...")

            # Load dataset
            print(f"    Loading dataset from {split_path}...")
            dataset = load_from_disk(split_path)
            num_examples = len(dataset)

            # Prepare output path
            output_path = os.path.join(args.output_dir, dataset_key)
            os.makedirs(output_path, exist_ok=True)

            # Tokenize with streaming processing using map with num_proc
            # The keep_in_memory=False and writer_batch_size help with memory efficiency
            print(f"    Tokenizing and saving to {output_path}...")
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                batch_size=args.batch_size,
                remove_columns=tokenizer_profile.tokenized_columns_to_remove(args.tokenizer_mode),
                num_proc=args.num_proc,
                keep_in_memory=False,  # Write to disk incrementally
                load_from_cache_file=False,  # Don't cache
                desc=f"Tokenizing {dataset_key}"
            )

            # Save immediately
            tokenized_dataset.save_to_disk(output_path)

            # Count tokens in a streaming fashion
            print(f"    Counting tokens...")
            total_tokens = 0

            # Iterate without loading all at once
            for i, example in enumerate(tokenized_dataset['input_ids']):
                total_tokens += len(example)
                if (i + 1) % 10000 == 0:
                    print(f"      Processed {i + 1:,} / {num_examples:,} examples...")

            avg_tokens = total_tokens / num_examples if num_examples > 0 else 0

            token_counts[dataset_key] = {
                'total_tokens': total_tokens,
                'num_examples': num_examples,
                'avg_tokens_per_example': avg_tokens
            }

            print(f"    Total tokens: {total_tokens:,}")
            print(f"    Number of examples: {num_examples:,}")
            print(f"    Average tokens per example: {avg_tokens:.2f}")

            # Free memory
            del dataset
            del tokenized_dataset

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
