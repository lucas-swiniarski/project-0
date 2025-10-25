import argparse

from .download_tinystories_dataset import download_tinystories
from .download_wikitext_dataset import download_wikitext
from .download_institutional_books_dataset import download_institutional_books

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
    args = parser.parse_args()

    download_tinystories(args.output_dir)
    download_wikitext(args.output_dir)
    download_institutional_books(args.output_dir,
                                 n_books=70000,
                                 n_val_books=100,
                                 n_test_books=100,
                                 max_threads=8,
                                 seed=42)
    print("\nAll pre-training datasets have been downloaded.")

if __name__ == "__main__":
    main()