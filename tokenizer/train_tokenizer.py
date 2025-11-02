import argparse
import glob
import os

from datasets import interleave_datasets, load_from_disk
from tokenizers import Tokenizer

import tokenizer.profiles as profiles


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
        default=32000,
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
    # TODO(swiniarski): Do something better than select what dataset to load partially.
    parser.add_argument(
        '--partial-dataset-name',
        type=str,
        default=None,
        help='Name of a dataset to partially load, identified by its directory name (e.g., "institutional-books-1.0").'
    )
    parser.add_argument(
        '--partial-dataset-fraction',
        type=float,
        default=None,
        help='Fraction of the partial dataset to use (e.g., 0.01 for 1%). Only used if --partial-dataset-name is set.'
    )

    args = parser.parse_args()
    
    # --- 1. Initialize a new tokenizer ---
    profile = profiles.TOKENIZER_NAME_TO_PROFILE[args.tokenizer_profile]()
    tokenizer = profile.create_tokenizer()

    # --- 2. Train the tokenizer ---
    print(f"Loading training dataset from {args.dataset_dir}...")
    train_dirs = glob.glob(os.path.join(args.dataset_dir, '*/train'))
    print(f"Searching for training datasets in subdirectories of {args.dataset_dir}...")
    # Find all 'train' subdirectories, e.g., /path/to/data/tinystories/train, /path/to/data/wikitext/train
    train_dirs = glob.glob(os.path.join(args.dataset_dir, '*/train'))
    if not train_dirs:
        raise FileNotFoundError(f"No 'train' subdirectories found in {args.dataset_dir}. Please check the path.")
    
    print(f"Found {len(train_dirs)} training datasets: {train_dirs}")
    # Load datasets and convert them to iterable datasets for streaming.
    # This is memory-efficient for large datasets.
    datasets = []
    for d in train_dirs:
        dataset = load_from_disk(d)
        if (args.partial_dataset_name and 
            args.partial_dataset_fraction and 
            args.partial_dataset_name in d):
            
            if not (0 < args.partial_dataset_fraction <= 1.0):
                raise ValueError("--partial-dataset-fraction must be between 0 and 1.")
            
            print(f"Sampling {args.partial_dataset_fraction:.2%} of dataset: {d}")
            # Use train_test_split to get a random fraction of the dataset.
            dataset = dataset.train_test_split(train_size=args.partial_dataset_fraction, shuffle=True, seed=42)['train']
        datasets.append(dataset)

    streaming_datasets = [d.to_iterable_dataset() for d in datasets]

    print(f"Training tokenizer with vocab size {args.vocab_size}...")
    trainer = profile.get_trainer(vocab_size=args.vocab_size)
    tokenizer.train_from_iterator(get_training_corpus(streaming_datasets), trainer=trainer, length=sum(d.num_rows for d in datasets))
    print("Training complete.")

    # --- 3. Save the tokenizer ---
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    tokenizer.save(args.output_path)
    print(f"Tokenizer saved to {args.output_path}")

if __name__ == "__main__":
    main()