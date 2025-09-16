import os
import argparse
from datasets import load_from_disk
from tokenizers import Tokenizer

def tokenize_function(examples, tokenizer):
    """
    Tokenizes a batch of text examples.
    We add Start Of Sentence ([SOS]) and End Of Sentence ([EOS]) tokens.
    """
    # The tokenizer expects a list of strings.
    output = tokenizer.encode_batch(examples["text"])
    
    # The output of encode_batch is a list of Encoding objects.
    # We extract the IDs and add SOS/EOS tokens.
    sos_token_id = tokenizer.token_to_id("[SOS]")
    eos_token_id = tokenizer.token_to_id("[EOS]")

    all_token_ids = []
    for encoding in output:
        all_token_ids.append([sos_token_id] + encoding.ids + [eos_token_id])

    return {"input_ids": all_token_ids}

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
    args = parser.parse_args()

    # --- 1. Load tokenizer and datasets ---
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    print(f"Loading datasets from {args.dataset_dir}...")
    raw_datasets = {
        split: load_from_disk(os.path.join(args.dataset_dir, split))
        for split in ["train", "validation", "test"]
    }

    # --- 2. Tokenize datasets ---
    print("Tokenizing datasets...")
    tokenized_datasets = {}
    for split, dataset in raw_datasets.items():
        print(f"Tokenizing {split} set...")
        tokenized_datasets[split] = dataset.map(
            tokenize_function,
            batched=True,
            fn_kwargs={'tokenizer': tokenizer},
            remove_columns=["text"], # We no longer need the raw text
            num_proc=args.num_proc
        )
        # --- 3. Save tokenized datasets ---
        output_path = os.path.join(args.output_dir, split)
        print(f"Saving tokenized {split} set to {output_path}...")
        tokenized_datasets[split].save_to_disk(output_path)

    print("Tokenization and saving complete.")

if __name__ == "__main__":
    main()
