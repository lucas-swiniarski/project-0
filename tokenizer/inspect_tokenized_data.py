import argparse
from datasets import load_from_disk
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

def main():
    parser = argparse.ArgumentParser(description="Interactively inspect tokenized data.")
    parser.add_argument(
        '--raw-dataset-dir',
        type=str,
        default='/home/lucas/data/v1/test',
        help='Path to the directory containing the raw text test set.'
    )
    parser.add_argument(
        '--tokenized-dataset-dir',
        type=str,
        default='/home/lucas/data/v1-tokenized/test',
        help='Path to the directory containing the tokenized test set.'
    )
    parser.add_argument(
        '--tokenizer-path',
        type=str,
        default='/home/lucas/tokenizer/v1/tokenizer.json',
        help='Path to the trained tokenizer file.'
    )
    args = parser.parse_args()

    # --- 1. Load tokenizer and datasets ---
    print("Loading datasets and tokenizer...")
    try:
        raw_dataset = load_from_disk(args.raw_dataset_dir)
        tokenized_dataset = load_from_disk(args.tokenized_dataset_dir)
        tokenizer = Tokenizer.from_file(args.tokenizer_path)

        # Set up the post-processor to correctly handle special tokens during decoding
        # This ensures [SOS] and [EOS] are handled correctly and subword tokens are joined.
        tokenizer.post_processor = TemplateProcessing(
            single="$A", # The main sequence. We can add special tokens to the template if we want to see them.
            special_tokens=[
                ("[SOS]", tokenizer.token_to_id("[SOS]")),
                ("[EOS]", tokenizer.token_to_id("[EOS]")),
            ],
        )
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Please ensure you have run the dataset preparation and tokenization scripts first.")
        return

    print(f"Successfully loaded. The test set has {len(raw_dataset)} examples.")
    print("Enter an index to inspect an example, or 'q' to quit.")

    # --- 2. Interactive loop ---
    while True:
        try:
            user_input = input(f"\nEnter index (0 to {len(raw_dataset) - 1}): ")
            if user_input.lower() == 'q':
                break

            index = int(user_input)
            if not (0 <= index < len(raw_dataset)):
                print(f"Index out of bounds. Please enter a number between 0 and {len(raw_dataset) - 1}.")
                continue

            # --- 3. Display data ---
            print("\n" + "="*20 + f" Example {index} " + "="*20)
            print(f"\n[Original Text]:\n'{raw_dataset[index]['text']}'")
            
            token_ids = tokenized_dataset[index]['input_ids']
            print(f"\n[Tokenized IDs]:\n{token_ids}")

            decoded_text = tokenizer.decode(token_ids)
            print(f"\n[Decoded Text (Clean)]:\n'{decoded_text}'")
            print("\n" + "="*55)

        except ValueError:
            print("Invalid input. Please enter an integer index or 'q'.")
        except KeyboardInterrupt:
            print("\nExiting.")
            break

if __name__ == "__main__":
    main()