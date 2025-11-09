"""
Convert a SentencePiece model (.model file) to HuggingFace tokenizers format (.json file).

This allows you to use the memory-efficient sentencepiece training pipeline
while still using HuggingFace tokenizers in your training code.
"""

import argparse
import os

import sentencepiece as spm
from tokenizers import Tokenizer, pre_tokenizers, decoders, processors
from tokenizers.models import Unigram


def convert_sentencepiece_to_hf(sp_model_path: str, output_path: str):
    """
    Convert a SentencePiece model to HuggingFace Tokenizer format.

    Args:
        sp_model_path: Path to the .model file from sentencepiece
        output_path: Path to save the HuggingFace tokenizer (.json)
    """
    # Load the sentencepiece model
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)

    print(f"Loaded SentencePiece model from {sp_model_path}")
    print(f"Vocabulary size: {sp.vocab_size()}")

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

    # Set up pre-tokenizer (Metaspace handles spaces like SentencePiece)
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()

    # Set up decoder (Metaspace to properly decode)
    tokenizer.decoder = decoders.Metaspace()

    # Define special tokens (matching train_sentencepiece.py)
    special_tokens = [
        "[UNK]",
        "[PAD]",
        "[SOS]",
        "[EOS]",
        "<SYSTEM>",
        "</SYSTEM>",
        "<USER>",
        "</USER>",
        "<MODEL>",
        "</MODEL>"
    ]

    # Set up post processor for special tokens
    special_token_map = [
        (token, tokenizer.token_to_id(token))
        for token in special_tokens
        if tokenizer.token_to_id(token) is not None
    ]

    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A",
        special_tokens=special_token_map,
    )

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

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.sp_model_path):
        raise FileNotFoundError(f"SentencePiece model not found: {args.sp_model_path}")

    # Convert
    convert_sentencepiece_to_hf(args.sp_model_path, args.output_path)

    print("\n" + "="*60)
    print("CONVERSION COMPLETE")
    print("="*60)
    print(f"\nYou can now use the tokenizer in your training pipeline:")
    print(f"  from tokenizers import Tokenizer")
    print(f"  tokenizer = Tokenizer.from_file('{args.output_path}')")


if __name__ == "__main__":
    main()
