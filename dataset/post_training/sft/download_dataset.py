import os
import argparse
from datasets import load_dataset

import tokenizer.profiles as tokenizer_profiles

def main():
    parser = argparse.ArgumentParser(description="Load, process, and save the Alpaca dataset for SFT.")
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/home/lucas/data/v1/raw/post_training/sft',
        help='The directory to save the processed datasets.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility of dataset splits.'
    )
    parser.add_argument(
        '--tokenizer-profile',
        type=str,
        default='post_training_v1',
        help='Random seed for reproducibility of dataset splits.'
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    tokenizer_profile = tokenizer_profiles.TOKENIZER_NAME_TO_PROFILE[args.tokenizer_profile]()
    os.makedirs(output_dir, exist_ok=True)

    # --- Load Dataset ---
    print("Loading yahma/alpaca-cleaned dataset...")
    dataset = load_dataset("yahma/alpaca-cleaned")

    # --- Split Dataset ---
    # The dataset only has a 'train' split. We need to create validation and test sets.
    print("Splitting dataset into train, validation, and test sets...")
    # Shuffle the dataset first for random splitting
    shuffled_dataset = dataset['train'].shuffle(seed=args.seed)

    # Create a 4k split for test+validation, then split that in half.
    test_valid_split = shuffled_dataset.train_test_split(test_size=4000, seed=args.seed)
    train_set = test_valid_split['train'] # The rest of the data
    
    # Split the 4k records into 2k for validation and 2k for test
    final_splits = test_valid_split['test'].train_test_split(test_size=0.5, seed=args.seed)
    validation_set = final_splits['train']
    test_set = final_splits['test']

    # --- Format Datasets ---
    print("Formatting datasets...")
    # Instantiate the profile to get access to its formatting method
    train_set = train_set.map(tokenizer_profile.format_example, 
                              fn_kwargs={'mode': 'alpaca'},
                              remove_columns=['instruction', 'input', 'output'])
    validation_set = validation_set.map(tokenizer_profile.format_example, 
                                        fn_kwargs={'mode': 'alpaca'},
                                        remove_columns=['instruction', 'input', 'output'])
    test_set = test_set.map(tokenizer_profile.format_example, 
                            fn_kwargs={'mode': 'alpaca'},
                            remove_columns=['instruction', 'input', 'output'])

    # --- Save Datasets to Disk ---
    print(f"Saving processed datasets to {output_dir}...")
    train_set.save_to_disk(os.path.join(output_dir, 'train'))
    validation_set.save_to_disk(os.path.join(output_dir, 'validation'))
    test_set.save_to_disk(os.path.join(output_dir, 'test'))

    print("Dataset processing and saving complete.")
    print(f"Train: {len(train_set)}, Validation: {len(validation_set)}, Test: {len(test_set)}")

if __name__ == "__main__":
    main()