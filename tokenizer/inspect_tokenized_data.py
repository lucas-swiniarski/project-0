import argparse
from datasets import load_from_disk
from tokenizers import Tokenizer
import tokenizer.profiles as tokenizer_profiles

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
    parser.add_argument(
        '--tokenizer-profile',
        type=str,
        default='post_training_v1',
        help='See tokenizer.profiles'
    )
    args = parser.parse_args()

    # --- 1. Load tokenizer and datasets ---
    print("Loading datasets and tokenizer...")
    tokenizer_profile = tokenizer_profiles.TOKENIZER_NAME_TO_PROFILE[args.tokenizer_profile]()
    try:
        raw_dataset = load_from_disk(args.raw_dataset_dir)
        tokenized_dataset = load_from_disk(args.tokenized_dataset_dir)
        tokenizer = Tokenizer.from_file(args.tokenizer_path)
        tokenizer = tokenizer_profile.configure_tokenizer(tokenizer)
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
            print(f"\n[Original Text]:\n'{raw_dataset[index]}'")
            
            token_ids = tokenized_dataset[index]['input_ids']
            print(f"\n[Tokenized Data]:\n{tokenized_dataset[index]}")

            decoded_text = tokenizer.decode(token_ids)
            print(f"\n[Decoded Text (Clean)]:\n'{decoded_text}'")

            decoded_with_special = tokenizer.decode(token_ids, skip_special_tokens=False)
            print(f"\n[Decoded Text (with special tokens)]:\n'{decoded_with_special}'")
            print("\n" + "="*55)
            
            tokens = [tokenizer.id_to_token(id) for id in token_ids]
            print(f"\n[Tokens (decoded one-by-one)]:\n{tokens}")
            print("\n" + "="*55)
            
        except ValueError:
            print("Invalid input. Please enter an integer index or 'q'.")
        except KeyboardInterrupt:
            print("\nExiting.")
            break

if __name__ == "__main__":
    main()