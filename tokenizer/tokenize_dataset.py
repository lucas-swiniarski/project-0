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
        default='/home/lucas/data/v1',
        help='Path to the directory containing the train, validation, and test sets.'
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
        default='/home/lucas/data/v1-tokenized',
        help='Directory to save the tokenized datasets.'
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
        default='pre_training_v1',
        help='Name of the tokenizer profile to use.'
    )
    parser.add_argument(
        '--tokenizer-mode',
        type=str,
        default='pre_training',
        help='Mode used for tokenizing, depends on the tokenizer profile.'
    )
    args = parser.parse_args()

    # --- 1. Load tokenizer and datasets ---
    print(f"Loading tokenizer from {args.tokenizer_path}, profile {args.tokenizer_profile} mode {args.tokenizer_mode}...")
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    tokenizer_profile = tokenizer_profiles.TOKENIZER_NAME_TO_PROFILE[args.tokenizer_profile]()
    tokenizer = tokenizer_profile.configure_tokenizer(tokenizer)

    print(f"Loading datasets from {args.dataset_dir}...")
    raw_datasets = {
        split: load_from_disk(os.path.join(args.dataset_dir, split))
        for split in ["train", "validation", "test"]
    }

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
    print("Tokenizing datasets...")
    tokenized_datasets = {}
        
    for split, dataset in raw_datasets.items():
        print(f"Tokenizing {split} set...")
        tokenized_datasets[split] = dataset.map(
            tokenize_function,  # Use the new helper function
            batched=True,
            remove_columns=tokenizer_profile.tokenized_columns_to_remove(args.tokenizer_mode),
            num_proc=args.num_proc
        )
        
        # --- 3. Save tokenized datasets ---
        output_path = os.path.join(args.output_dir, split)
        print(f"Saving tokenized {split} set to {output_path}...")
        tokenized_datasets[split].save_to_disk(output_path)

    print("Tokenization and saving complete.")

if __name__ == "__main__":
    main()
