import argparse
import os

from datasets import load_dataset
from .download_dataset_utils import write_datasets

def download_tinystories(output_dir: str):
    """
    Downloads the TinyStories dataset and saves it to the specified directory.

    Args:
        output_dir (str): The directory to save the dataset to.
    """
    print("Downloading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories")

    write_datasets({
        "train": dataset["train"],
        "validation": dataset["validation"],
    }, output_dir, "tinystories")
    
    print("TinyStories dataset downloaded and saved successfully.")


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description="Download and process the TinyStories dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="The directory where the dataset will be saved.",
    )
    args = parser.parse_args()
    download_tinystories(args.output_dir)

if __name__ == "__main__":
    main()