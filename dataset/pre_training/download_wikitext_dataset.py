import argparse
import os
import re

from datasets import Dataset, load_dataset

from .download_dataset_utils import write_datasets


def download_wikitext(output_dir: str, num_proc: int):
    """
    Downloads the WikiText-103-raw-v1 dataset and saves it to the specified directory.

    Args:
        output_dir (str): The directory to save the dataset to.
        num_proc (int): Number of threads.
    """
    print("Downloading WikiText-103-raw-v1 dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

    # --- Process and Combine Datasets ---
    print("Processing wikitext-2 dataset to coalesce articles...")
    coalesced_wikitext_train = coalesce_wikitext_articles(dataset["train"])
    coalesced_wikitext_validation = coalesce_wikitext_articles(dataset["validation"])
    coalesced_wikitext_test = coalesce_wikitext_articles(dataset["test"])
    
    write_datasets({
        "train": coalesced_wikitext_train,
        "validation": coalesced_wikitext_validation,
        "test": coalesced_wikitext_test,
    }, output_dir, "wikitext-103-raw-v1", num_proc=num_proc)

    print("WikiText-103-raw-v1 dataset downloaded and saved successfully.")


def coalesce_wikitext_articles(dataset_split):
    """
    Coalesce dataset of the same Wikipedia article into a single string.
    New articles are identified by titles like ' = {title} = \n'.
    """
    articles = []
    current_article_lines = []
    # Regex to find titles like ' = title = ' but not ' == subtitle == '
    title_pattern = re.compile(r"^\s=\s[^=].*[^=]\s=\s\n$")

    for item in dataset_split:
        line = item["text"]
        # We check for titles that are not subtitles to delimit articles.
        # The first article does not start with a title, so we also check if current_article_lines is not empty.
        if title_pattern.match(line) and current_article_lines:
            articles.append("".join(current_article_lines).strip())
            current_article_lines = [line]
        else:
            # We only add non-empty lines to avoid too many blank lines inside articles.
            if line.strip():
                current_article_lines.append(line)

    # Add the last article
    if current_article_lines:
        articles.append("".join(current_article_lines).strip())

    return Dataset.from_dict({"text": articles})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and process the WikiText-103-raw-v1 dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="The directory where the dataset will be saved.",
    )
    args = parser.parse_args()
    download_wikitext(args.output_dir)
