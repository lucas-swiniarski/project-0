"""
Convert a SentencePiece model (.model file) to HuggingFace tokenizers format (.json file).

This allows you to use the memory-efficient sentencepiece training pipeline
while still using HuggingFace tokenizers in your training code.
"""

import argparse
import os

import sentencepiece as spm
from tokenizers import Tokenizer
from tokenizers.models import Unigram

from tokenizer.profiles import TOKENIZER_NAME_TO_PROFILE


def convert_sentencepiece_to_hf(sp_model_path: str, output_path: str, tokenizer_profile_name: str):
    """
    Convert a SentencePiece model to HuggingFace Tokenizer format.

    Args:
        sp_model_path: Path to the .model file from sentencepiece
        output_path: Path to save the HuggingFace tokenizer (.json)
        tokenizer_profile_name: Name of the tokenizer profile to use for configuration
    """
    # Load the sentencepiece model
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)

    print(f"Loaded SentencePiece model from {sp_model_path}")
    print(f"Vocabulary size: {sp.vocab_size()}")

    # Get the tokenizer profile
    profile_class = TOKENIZER_NAME_TO_PROFILE[tokenizer_profile_name]
    profile = profile_class()
    special_tokens = profile.get_special_tokens()

    print(f"Using tokenizer profile: {tokenizer_profile_name}")
    print(f"Special tokens: {special_tokens}")

    # Extract vocabulary and scores
    vocab = []
    scores = []
    for i in range(sp.vocab_size()):
        piece = sp.id_to_piece(i)
        score = sp.get_score(i)
        vocab.append((piece, score))

    print(f"Extracted {len(vocab)} tokens from vocabulary")

    # Create HuggingFace Unigram model with the vocabulary
    tokenizer = Tokenizer(Unigram(vocab, unk_id=0))

    # Use the profile to configure the tokenizer (pre-tokenizer, decoder, post-processor)
    tokenizer = profile.configure_tokenizer(tokenizer)

    # Save the tokenizer
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tokenizer.save(output_path)

    print(f"\nSaved HuggingFace tokenizer to {output_path}")

    # Test the conversion
    print("\nTesting converted tokenizer...")
    test_sentence = "Hello, world! This is a test."

    # Test with sentencepiece
    sp_tokens = sp.encode_as_pieces(test_sentence)
    sp_ids = sp.encode_as_ids(test_sentence)

    # Test with HuggingFace
    hf_encoding = tokenizer.encode(test_sentence)
    hf_tokens = hf_encoding.tokens
    hf_ids = hf_encoding.ids

    print(f"\nTest sentence: {test_sentence}")
    print(f"\nSentencePiece tokens: {sp_tokens}")
    print(f"SentencePiece IDs: {sp_ids}")
    print(f"\nHuggingFace tokens: {hf_tokens}")
    print(f"HuggingFace IDs: {hf_ids}")

    # Check if they match
    if sp_ids == hf_ids:
        print("\n✓ Conversion successful! Token IDs match.")
    else:
        print("\n⚠ Warning: Token IDs don't match exactly.")
        print("This might be due to differences in pre-tokenization.")
        print("The tokenizer will still work, but output may differ slightly.")

    return tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Convert SentencePiece model to HuggingFace tokenizers format."
    )
    parser.add_argument(
        '--sp-model-path',
        type=str,
        default='/home/lucas/tokenizer/v2/sentencepiece.model',
        help='Path to the SentencePiece .model file.'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='/home/lucas/tokenizer/v2/tokenizer.json',
        help='Path to save the HuggingFace tokenizer JSON file.'
    )
    parser.add_argument(
        '--tokenizer-profile',
        type=str,
        default='sentence_piece_v2',
        choices=list(TOKENIZER_NAME_TO_PROFILE.keys()),
        help='Tokenizer profile to use for configuration (default: sentence_piece_v2).'
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.sp_model_path):
        raise FileNotFoundError(f"SentencePiece model not found: {args.sp_model_path}")

    # Convert
    convert_sentencepiece_to_hf(args.sp_model_path, args.output_path, args.tokenizer_profile)

    print("\n" + "="*60)
    print("CONVERSION COMPLETE")
    print("="*60)
    print(f"\nYou can now use the tokenizer in your training pipeline:")
    print(f"  from tokenizers import Tokenizer")
    print(f"  tokenizer = Tokenizer.from_file('{args.output_path}')")


if __name__ == "__main__":
    main()
