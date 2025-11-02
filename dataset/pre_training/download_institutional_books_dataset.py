import argparse
import concurrent.futures
import glob
import json
import os
import queue
import re
import threading
from itertools import islice
from typing import Callable

from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm

from .download_dataset_utils import write_datasets


def download_institutional_books(
    output_dir: str,
    n_train_books: int = 70000,
    n_val_books: int = 100,
    n_test_books: int = 100,
    num_proc: int = 8,
    seed: int = 42,
    books_per_file: int = 100,
):
    """Download institutional/institutional-books-1.0

    Args:
        output_dir (str): The directory to save the dataset to.
        n_train_books (int): The number of books to download (1 book ~ 150k o200k tokens).
        n_val_books (int): The number of books for the validation set.
        n_test_books (int): The number of books for the test set.
        num_proc (int): The maximum number of threads to use for processing.
        seed (int): Random seed for shuffling.
        books_per_file (int): Number of books to save in each intermediate JSONL file.
    """
    print("Downloading institutional/institutional-books-1.0 dataset...")
    dataset = load_dataset(
        "institutional/institutional-books-1.0", streaming=True, split="train"
    )
    shuffled_dataset = dataset.shuffle(seed=seed)
    total_books_to_process = n_train_books + n_val_books + n_test_books
    print(
        f"Processing {total_books_to_process} books in total: {n_train_books} for train, {n_val_books} for validation, {n_test_books} for test."
    )

    # Create an iterator from the shuffled dataset
    dataset_iterator = iter(shuffled_dataset)

    # Use a persistent directory for intermediate JSONL files
    jsonl_output_dir = os.path.join(output_dir, "institutional-books-1.0-jsonl")
    os.makedirs(jsonl_output_dir, exist_ok=True)
    print(f"Using intermediate directory for JSONL files: {jsonl_output_dir}")

    splits_to_process = {
        "train": n_train_books,
        "validation": n_val_books,
        "test": n_test_books,
    }

    for split_name, n_books in splits_to_process.items():
        print(f"--- Processing {split_name} split ---")
        split_dir = os.path.join(jsonl_output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        # --- Resumption Logic ---
        processed_books = 0
        existing_files = glob.glob(os.path.join(split_dir, f"{split_name}_*.jsonl"))
        if existing_files:
            # Assume full files have `books_per_file` books.
            processed_books = len(existing_files) * books_per_file
            print(f"Found {len(existing_files)} existing files, resuming download. Already processed ~{processed_books} books.")

        if processed_books >= n_books:
            print(f"Already have {processed_books} books for {split_name}. Skipping.")
            continue

        # Skip already processed books in the dataset iterator
        print(f"Skipping {processed_books} books in the dataset iterator...")
        for _ in tqdm(islice(dataset_iterator, processed_books), total=processed_books, desc="Skipping books"):
            pass
        
        books_to_process_for_split = n_books - processed_books
        print(f"Processing {books_to_process_for_split} more books for {split_name} split.")

        processor = SampleIteratorJob(
            stream=dataset_iterator,
            sample_processor=process_sample,
            output_dir=split_dir,
            file_prefix=split_name,
            total_samples=books_to_process_for_split,
            samples_per_file=books_per_file,
            start_file_index=len(existing_files),
            max_workers=num_proc,
        )
        processor.run()
        print(f"Successfully processed {processor.writer_success_count} additional books for {split_name}.")

    # --- Final Conversion to Arrow Format ---
    print("\nConverting all JSONL files to final Arrow dataset format...")
    all_splits_data = {}
    for split_name in splits_to_process:
        split_dir = os.path.join(jsonl_output_dir, split_name)
        jsonl_files = sorted(glob.glob(os.path.join(split_dir, f"{split_name}_*.jsonl")))
        if jsonl_files:
            print(f"Loading {len(jsonl_files)} files for {split_name} split...")
            all_splits_data[split_name] = Dataset.from_json(jsonl_files)

    if all_splits_data:
        write_datasets(all_splits_data, output_dir, "institutional-books-1.0", num_proc=num_proc)
    else:
        print("No data was processed. Skipping final dataset creation.")

    print(
        "Institutional/institutional-books-1.0 dataset downloaded and saved successfully."
    )


class SampleIteratorJob:
    def __init__(
        self,
        stream,
        sample_processor: Callable[[dict], tuple[str, int]],
        output_dir: str,
        file_prefix: str,
        total_samples: int,
        samples_per_file: int,
        start_file_index: int = 0,
        max_workers: int = 8,
    ):
        self.stream = stream
        self.sample_processor = sample_processor
        self.output_dir = output_dir
        self.file_prefix = file_prefix
        self.max_workers = max_workers
        self.total_samples = total_samples

        self.processed_samples_queue = queue.Queue()
        self.lock = threading.Lock()
        self.writer_success_count = 0
        self.writer_failed_count = 0
        self.total_tokens_processed = 0
        self.samples_per_file = samples_per_file
        self.start_file_index = start_file_index

    def _writer(self):
        """Writer thread that consumes processed samples and writes to chunked JSONL files."""
        file_index = self.start_file_index
        samples_in_current_file = 0
        f = None

        while True:
            try:
                text, n_tokens = self.processed_samples_queue.get()
                if text is None:  # Sentinel value to stop
                    if f:
                        f.close()
                    self.processed_samples_queue.task_done()
                    break

                if f is None or samples_in_current_file >= self.samples_per_file:
                    if f:
                        f.close()
                    file_path = os.path.join(self.output_dir, f"{self.file_prefix}_{file_index:05d}.jsonl")
                    f = open(file_path, "w", encoding="utf-8")
                    file_index += 1
                    samples_in_current_file = 0

                f.write(json.dumps({"text": text}) + "\n")
                samples_in_current_file += 1
                with self.lock:
                    self.writer_success_count += 1
                    self.total_tokens_processed += n_tokens
            except Exception as exc:
                with self.lock:
                    self.writer_failed_count += 1
                print(f"Writer generated an exception: {exc}")
            self.processed_samples_queue.task_done()

    def _sample_processor(self, sample):
        text, n_tokens = self.sample_processor(sample)
        self.processed_samples_queue.put((text, n_tokens))
            
    def run(self):
        writer_thread = threading.Thread(target=self._writer)
        writer_thread.start()
        
        dataset_stream = islice(self.stream, self.total_samples)

        with tqdm(total=self.total_samples, desc="Processing books", ncols=120) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for i, sample in enumerate(dataset_stream):
                    executor.submit(self._sample_processor, sample)
                    pbar.update(1)
                    pbar.set_postfix(suc=self.writer_success_count,tok=f'{self.total_tokens_processed/1e6:.1f}M')

        self.processed_samples_queue.put((None, None)) # Sentinel value
        self.processed_samples_queue.join()
        writer_thread.join()

def process_sample(sample: dict[str, str]) -> tuple[str, int]:
    """Process one sample of institutional books into a string and n_tokens.

    Args:
        sample (dict[str, str]): The sample to process.

    Return:
        The extracted text and estimated number of tokens using o200k tokenizer.
    """
    text = f"{sample['title_src']} by {sample['author_src']}, published {sample['date1_src']}\n"

    for page in sample["text_by_page_src"]:
        text += page

    return text, sample["token_count_o200k_base_gen"]


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description="Download and process the institutional/institutional-books-1.0 dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="The directory where the dataset will be saved.",
    )
    parser.add_argument(
        "--n-train-books",
        type=int,
        default=70000,
        help="The number of books to download.",
    )
    parser.add_argument(
        "--n-val-books",
        type=int,
        default=50,
        help="The number of books for the validation set.",
    )
    parser.add_argument(
        "--n-test-books",
        type=int,
        default=50,
        help="The number of books for the test set.",
    )
    parser.add_argument(
        "--books-per-file",
        type=int,
        default=100,
        help="Number of books per intermediate JSONL file.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=8,
        help="The maximum number of threads to use for processing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for shuffling the dataset.",
    )
    args = parser.parse_args()
    download_institutional_books(
        args.output_dir,
        args.n_train_books,
        args.n_val_books,
        args.n_test_books,
        args.num_proc,
        args.seed,
        args.books_per_file,
    )


if __name__ == "__main__":
    main()
