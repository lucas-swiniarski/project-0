import argparse
import concurrent.futures
import json
import os
import queue
import tempfile
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from itertools import islice
from typing import Callable

from datasets import Dataset, load_dataset
from tqdm import tqdm

from .download_dataset_utils import write_datasets


def download_institutional_books(
    output_dir: str,
    n_train_books: int = 70000,
    n_val_books: int = 100,
    n_test_books: int = 100,
    max_threads: int = 8,
    seed: int = 42,
):
    """Download institutional/institutional-books-1.0

    Args:
        output_dir (str): The directory to save the dataset to.
        n_train_books (int): The number of books to download (1 book ~ 150k o200k tokens).
        n_val_books (int): The number of books for the validation set.
        n_test_books (int): The number of books for the test set.
        max_threads (int): The maximum number of threads to use for processing.
        seed (int): Random seed for shuffling.
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

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Using temporary directory: {tmpdir}")

        # Process each split and write to temporary JSONL files
        print("Processing train...")
        train_output_file = os.path.join(tmpdir, "train.jsonl")
        processor = SampleIteratorJob(
            dataset_iterator, process_sample, train_output_file, n_train_books, max_workers=max_threads
        )
        processor.run()
        print(f"Successfully processed {processor.writer_success_count} books for train.")
        
        print("Processing validation...")
        validation_output_file = os.path.join(tmpdir, "val.jsonl")
        processor = SampleIteratorJob(
            dataset_iterator, process_sample, validation_output_file, n_val_books, max_workers=max_threads
        )
        processor.run()
        print(f"Successfully processed {processor.writer_success_count} books for validation.")
        
        print("Processing test...")
        test_output_file = os.path.join(tmpdir, "test.jsonl")
        processor = SampleIteratorJob(
            dataset_iterator, process_sample, test_output_file, n_test_books, max_workers=max_threads
        )
        processor.run()
        print(f"Successfully processed {processor.writer_success_count} books for test.")

        # Load from JSONL and save to disk in Arrow format, apparently from_json streams ?
        print("Converting temporary files to final dataset format...")
        train_dataset = Dataset.from_json(train_output_file)
        val_dataset = Dataset.from_json(validation_output_file)
        test_dataset = Dataset.from_json(test_output_file)
        val_dataset = val_dataset.rename_column('split', 'validation') # from_json does not have a split argument
        test_dataset = test_dataset.rename_column('split', 'test')

        write_datasets(
            {
                "train": train_dataset,
                "validation": val_dataset,
                "test": test_dataset,
            },
            output_dir,
            "institutional-books-1.0",
        )

    print(
        "Institutional/institutional-books-1.0 dataset downloaded and saved successfully."
    )


class SampleIteratorJob:
    def __init__(
        self,
        stream,
        sample_processor: Callable[[dict[str, str]], tuple[str, int]],
        tmpfile: str,
        total_samples: int,
        max_workers: int = 8,
    ):
        self.stream = stream
        self.sample_processor = sample_processor
        self.tmpfile = tmpfile
        self.max_workers = max_workers
        self.total_samples = total_samples

        self.processed_samples_queue = queue.Queue()
        self.lock = threading.Lock()
        self.writer_success_count = 0
        self.writer_failed_count = 0
        self.total_tokens_processed = 0

    def _writer(self):
        """Writer thread that consumes processed samples and writes to a JSONL file."""
        with open(self.tmpfile, "w", encoding="utf-8") as f:
            while True:
                try:
                    text, n_tokens = self.processed_samples_queue.get()
                    if text is None:
                        # Signal that we're done writing.
                        self.processed_samples_queue.task_done()
                        break
                    f.write(json.dumps({"text": text}) + "\n")
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
                for sample in dataset_stream:
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
        "--max-threads",
        type=int,
        default=3,
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
        args.max_threads,
        args.seed,
    )


if __name__ == "__main__":
    main()
