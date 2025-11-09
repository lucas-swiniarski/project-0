"""
Merges sharded datasets from the pre-training directory into a single text file
for training with the sentencepiece library.

This script reads datasets from /home/lucas/data/v2/raw/pre_training/{dataset_name}/train/
and writes all text samples to a single output file, with optional subsampling for memory efficiency.
"""

import argparse
import glob
import os
import random

from datasets import load_from_disk

from tokenizer.train_tokenizer_configurations import DATASET_CONFIGURATIONS


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a corpus file for sentencepiece training from sharded datasets."
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='/home/lucas/data/v2/raw/pre_training/',
        help='Path to the training dataset directory.'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='/home/lucas/data/v2/sentencepiece_corpus.txt',
        help='Path to the output corpus text file.'
    )
    parser.add_argument(
        '--dataset-configuration',
        type=str,
        default='medium',
        help=f'Dataset configuration to use. Available: {", ".join(DATASET_CONFIGURATIONS.keys())}'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to include (for memory efficiency). If not set, includes all.'
    )
    parser.add_argument(
        '--shuffle',
        action='store_true',
        help='Shuffle samples before writing (useful with --max-samples for representative sampling).'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for shuffling (default: 42).'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=None,
        help='Split long texts into chunks of this many words. Useful for books. If not set, keeps original samples intact.'
    )
    parser.add_argument(
        '--max-lines',
        type=int,
        default=None,
        help='Maximum number of lines to write to output file. Applied after chunking and shuffling. Use this to limit memory during tokenizer training.'
    )

    args = parser.parse_args()

    # Validate dataset configuration
    if args.dataset_configuration not in DATASET_CONFIGURATIONS:
        raise ValueError(
            f"Unknown dataset configuration: {args.dataset_configuration}. "
            f"Available configurations: {', '.join(DATASET_CONFIGURATIONS.keys())}"
        )

    config = DATASET_CONFIGURATIONS[args.dataset_configuration]
    print(f"Using dataset configuration: {args.dataset_configuration}")
    print(f"Configuration: {config}")

    # Set random seed
    if args.shuffle:
        random.seed(args.seed)

    # Load datasets
    print(f"Loading datasets from {args.dataset_dir}...")
    datasets = []
    total_samples = 0

    for dataset_name, num_shards in config.items():
        if num_shards == 0:
            print(f"Skipping {dataset_name} (0 shards requested)")
            continue

        dataset_path = os.path.join(args.dataset_dir, dataset_name, 'train')

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Dataset path not found: {dataset_path}. "
                f"Please ensure {dataset_name} exists in {args.dataset_dir}"
            )

        # Load the full dataset
        dataset = load_from_disk(dataset_path)

        # Get all available shards for this dataset
        shard_files = sorted(glob.glob(os.path.join(dataset_path, 'data-*.arrow')))
        available_shards = len(shard_files)

        if num_shards > available_shards:
            print(
                f"Warning: Requested {num_shards} shards for {dataset_name}, "
                f"but only {available_shards} available. Using all {available_shards} shards."
            )
            num_shards = available_shards

        # Calculate fraction to load based on shard count
        if num_shards < available_shards:
            fraction = num_shards / available_shards
            print(f"Loading {num_shards}/{available_shards} shards ({fraction:.2%}) of dataset: {dataset_name}")
            dataset = dataset.train_test_split(train_size=fraction, shuffle=True, seed=args.seed)['train']
        else:
            print(f"Loading all {available_shards} shards of dataset: {dataset_name}")

        datasets.append(dataset)
        total_samples += len(dataset)
        print(f"  Loaded {len(dataset):,} samples from {dataset_name}")

    print(f"\nTotal samples loaded: {total_samples:,}")

    # Determine final sample count
    if args.max_samples and args.max_samples < total_samples:
        final_sample_count = args.max_samples
        print(f"Will subsample to {final_sample_count:,} samples")
    else:
        final_sample_count = total_samples
        if args.max_samples:
            print(f"--max-samples ({args.max_samples:,}) is larger than total samples, using all samples")

    # Collect all text samples
    print("\nCollecting text samples...")
    all_samples = []
    for dataset in datasets:
        for example in dataset:
            text = example['text']

            # Split into chunks if requested
            if args.chunk_size:
                words = text.split()
                for i in range(0, len(words), args.chunk_size):
                    chunk = ' '.join(words[i:i + args.chunk_size])
                    if chunk:  # Skip empty chunks
                        all_samples.append(chunk)
            else:
                all_samples.append(text)

    # Shuffle if requested
    if args.shuffle:
        print(f"Shuffling samples with seed {args.seed}...")
        random.shuffle(all_samples)

    # Subsample if requested
    if args.max_samples and args.max_samples < len(all_samples):
        all_samples = all_samples[:args.max_samples]
        print(f"Subsampled to {len(all_samples):,} samples")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Write to file
    print(f"\nWriting corpus to {args.output_file}...")
    samples_written = 0
    max_lines_to_write = args.max_lines if args.max_lines else len(all_samples)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            # Stop if we've reached max_lines limit
            if samples_written >= max_lines_to_write:
                print(f"  Reached max_lines limit of {args.max_lines:,}, stopping...")
                break

            # Clean the text: remove extra whitespace, ensure single line
            cleaned = ' '.join(sample.split())
            if cleaned:  # Skip empty samples
                f.write(cleaned + '\n')
                samples_written += 1

            # Progress indicator
            if samples_written % 100000 == 0:
                print(f"  Written {samples_written:,} samples...")

    print(f"\nDone! Written {samples_written:,} samples to {args.output_file}")

    # Print file size
    file_size_mb = os.path.getsize(args.output_file) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")


if __name__ == "__main__":
    main()
