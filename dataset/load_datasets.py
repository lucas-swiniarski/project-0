import os
import argparse
from datasets import load_dataset, concatenate_datasets
from .wikipedia_dataset_utils import coalesce_wikitext_articles

def main():
    parser = argparse.ArgumentParser(description="Load, process, and save datasets.")
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/home/lucas/data/v0',
        help='The directory to save the processed datasets.'
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # --- Load Datasets ---
    print("Loading TinyStories and wikitext-2 datasets...")
    tinystories_dataset = load_dataset('roneneldan/TinyStories')
    wikitext2_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

    # --- Process and Combine Datasets ---
    print("Processing wikitext-2 dataset to coalesce articles...")
    coalesced_wikitext_train = coalesce_wikitext_articles(wikitext2_dataset['train'])
    coalesced_wikitext_validation = coalesce_wikitext_articles(wikitext2_dataset['validation'])
    coalesced_wikitext_test = coalesce_wikitext_articles(wikitext2_dataset['test'])

    print("Combining TinyStories and processed wikitext-2 datasets...")
    combined_train_set = concatenate_datasets([tinystories_dataset['train'], coalesced_wikitext_train])
    combined_validation_set = concatenate_datasets([tinystories_dataset['validation'], coalesced_wikitext_validation])
    # The test set will only be from the processed wikitext data as TinyStories doesn't have one.
    combined_test_set = coalesced_wikitext_test

    # --- Save Datasets to Disk ---
    print(f"Saving processed datasets to {output_dir}...")
    combined_train_set.save_to_disk(os.path.join(output_dir, 'train'))
    combined_validation_set.save_to_disk(os.path.join(output_dir, 'validation'))
    combined_test_set.save_to_disk(os.path.join(output_dir, 'test'))

    print("Dataset processing and saving complete.")

if __name__ == "__main__":
    main()