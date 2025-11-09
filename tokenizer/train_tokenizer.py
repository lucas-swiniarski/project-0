import argparse
import glob
import os

from datasets import interleave_datasets, load_from_disk
from tokenizers import Tokenizer

import tokenizer.profiles as profiles
from tokenizer.train_tokenizer_configurations import DATASET_CONFIGURATIONS


def get_training_corpus(datasets):
    """
    A generator function to yield text samples one by one from a list of datasets.
    This is memory-efficient as it processes datasets in a streaming fashion.

    Args:
        datasets (list): A list of Hugging Face Dataset objects to iterate over.
    """
    dataset_iter = iter(interleave_datasets(datasets))
    for example in dataset_iter:
        yield example['text']

def main():
    parser = argparse.ArgumentParser(description="Train a new BPE tokenizer from scratch.")
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='/home/lucas/data/v2/raw/pre_training/',
        help='Path to the training dataset directory.'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=64000,
        help='The desired vocabulary size for the tokenizer.'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='/home/lucas/tokenizer/v1/tokenizer.json',
        help='Path where the trained tokenizer will be saved.'
    )
    parser.add_argument(
        '--tokenizer-profile',
        type=str,
        default='sentence_piece_v2',
        help='See tokenizer.profiles.'
    )
    parser.add_argument(
        '--dataset-configuration',
        type=str,
        default='medium',
        help=f'Dataset configuration to use. Available configurations: {", ".join(DATASET_CONFIGURATIONS.keys())}'
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

    # --- 1. Initialize a new tokenizer ---
    profile = profiles.TOKENIZER_NAME_TO_PROFILE[args.tokenizer_profile]()
    tokenizer = profile.create_tokenizer()

    # --- 2. Train the tokenizer ---
    print(f"Loading training datasets from {args.dataset_dir}...")

    datasets = []
    for dataset_name, num_shards in config.items():
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
            dataset = dataset.train_test_split(train_size=fraction, shuffle=True, seed=42)['train']
        else:
            print(f"Loading all {available_shards} shards of dataset: {dataset_name}")

        datasets.append(dataset)

    streaming_datasets = [d.to_iterable_dataset() for d in datasets]

    print(f"Training tokenizer with vocab size {args.vocab_size}...")
    trainer = profile.get_trainer(vocab_size=args.vocab_size)
    tokenizer.train_from_iterator(
        get_training_corpus(streaming_datasets), 
        trainer=trainer, 
        length=sum(d.num_rows for d in datasets))
    print("Training complete.")

    # --- 3. Save the tokenizer ---
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    tokenizer.save(args.output_path)
    print(f"Tokenizer saved to {args.output_path}")

if __name__ == "__main__":
    main()
