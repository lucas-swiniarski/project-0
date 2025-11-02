import argparse
import os

from .download_institutional_books_dataset import download_institutional_books
from .download_tinystories_dataset import download_tinystories
from .download_wikitext_dataset import download_wikitext


def main():
    """
    Main function to download all pre-training datasets.
    """
    parser = argparse.ArgumentParser(
        description="Download all pre-training datasets (TinyStories and WikiText)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="The root directory where the datasets will be saved.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=8,
        help="Number of threads.",
    )
    args = parser.parse_args()
    print(f'Downloading pre-training datasets to {args.output_dir}...')
    os.makedirs(args.output_dir, exist_ok=True)
    download_tinystories(args.output_dir, args.num_proc)
    download_wikitext(args.output_dir, args.num_proc)    
    download_institutional_books(args.output_dir,
                                 n_books=70000,
                                 n_val_books=100,
                                 n_test_books=100,
                                 num_proc=args.num_proc,
                                 seed=1)
    print("\nAll pre-training datasets have been downloaded.")

if __name__ == "__main__":
    main()