"""
Train a SentencePiece tokenizer directly using the sentencepiece library.

This approach is more memory-efficient than train_from_iterator because:
1. It can subsample the input with input_sentence_size
2. It supports multi-threading
3. It reads from disk instead of loading everything into memory
"""

import argparse
import os

import sentencepiece as spm

from tokenizer.profiles import TOKENIZER_NAME_TO_PROFILE


def main():
    parser = argparse.ArgumentParser(
        description="Train a SentencePiece tokenizer from a corpus file."
    )
    parser.add_argument(
        '--input-file',
        type=str,
        default='/home/lucas/data/v2/raw/pre_training/sentencepiece_corpus.txt',
        help='Path to the input corpus text file.'
    )
    parser.add_argument(
        '--model-prefix',
        type=str,
        default='/home/lucas/tokenizer/v2/sentencepiece',
        help='Prefix for output model files (will create .model and .vocab files).'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=64000,
        help='The desired vocabulary size for the tokenizer.'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='unigram',
        choices=['unigram', 'bpe', 'char', 'word'],
        help='Model algorithm: unigram (default), bpe, char, or word.'
    )
    parser.add_argument(
        '--input-sentence-size',
        type=int,
        default=10000000,
        help='Maximum number of sentences to use for training (subsamples if larger). Default: 10M.'
    )
    parser.add_argument(
        '--shuffle-input-sentence',
        action='store_true',
        default=True,
        help='Shuffle input sentences before training (recommended with subsampling).'
    )
    parser.add_argument(
        '--num-threads',
        type=int,
        default=16,
        help='Number of threads to use for training (default: 16).'
    )
    parser.add_argument(
        '--character-coverage',
        type=float,
        default=0.9995,
        help='Character coverage for the tokenizer (default: 0.9995).'
    )
    parser.add_argument(
        '--normalization-rule-name',
        type=str,
        default='nfkc',
        choices=['nmt_nfkc', 'nfkc', 'nmt_nfkc_cf', 'nfkc_cf', 'identity'],
        help='Normalization rule (default: nfkc).'
    )
    parser.add_argument(
        '--max-sentence-length',
        type=int,
        default=16384,
        help='Maximum sentence length in bytes. Lines longer than this are skipped. For books, use 4194304 (4MB). Default: 16384.'
    )
    parser.add_argument(
        '--tokenizer-profile',
        type=str,
        default='sentence_piece_v2',
        choices=list(TOKENIZER_NAME_TO_PROFILE.keys()),
        help='Tokenizer profile to use for special tokens (default: sentence_piece_v2).'
    )

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    # Ensure output directory exists
    output_dir = os.path.dirname(args.model_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Get special tokens from the selected profile
    profile_class = TOKENIZER_NAME_TO_PROFILE[args.tokenizer_profile]
    profile = profile_class()
    special_tokens = profile.get_special_tokens()

    print(f"Training SentencePiece tokenizer...")
    print(f"  Input file: {args.input_file}")
    print(f"  Model prefix: {args.model_prefix}")
    print(f"  Vocab size: {args.vocab_size}")
    print(f"  Model type: {args.model_type}")
    print(f"  Tokenizer profile: {args.tokenizer_profile}")
    print(f"  Special tokens: {special_tokens}")
    print(f"  Max input sentences: {args.input_sentence_size:,}")
    print(f"  Threads: {args.num_threads}")
    print(f"  Character coverage: {args.character_coverage}")
    print(f"  Max sentence length: {args.max_sentence_length:,} bytes")

    # Train the model
    spm.SentencePieceTrainer.train(
        input=args.input_file,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,

        # Memory efficiency parameters
        input_sentence_size=args.input_sentence_size,
        shuffle_input_sentence=args.shuffle_input_sentence,

        # Performance parameters
        num_threads=args.num_threads,

        # Special tokens
        user_defined_symbols=special_tokens,
        unk_id=0,  # [UNK]
        pad_id=1,  # [PAD]
        bos_id=2,  # [SOS]
        eos_id=3,  # [EOS]

        # Character coverage
        character_coverage=args.character_coverage,

        # Normalization
        normalization_rule_name=args.normalization_rule_name,

        # Other useful parameters
        split_by_whitespace=True,
        split_by_unicode_script=True,
        split_by_number=True,
        split_digits=True,
        byte_fallback=True,  # Handle any character, even if not in training data

        # Training parameters
        num_sub_iterations=2,
        max_sentence_length=args.max_sentence_length,
        seed_sentencepiece_size=1000000,
        shrinking_factor=0.75,

        # Progress
        train_extremely_large_corpus=False,
    )

    print(f"\nTraining complete!")
    print(f"Model saved to: {args.model_prefix}.model")
    print(f"Vocabulary saved to: {args.model_prefix}.vocab")

    # Test the tokenizer
    print("\nTesting tokenizer...")
    sp = spm.SentencePieceProcessor()
    sp.load(f"{args.model_prefix}.model")

    test_sentences = [
        "Hello, world!",
        "This is a test of the SentencePiece tokenizer.",
        "<USER>What is 2+2?</USER><MODEL>2+2 equals 4.</MODEL>",
    ]

    for sentence in test_sentences:
        tokens = sp.encode_as_pieces(sentence)
        ids = sp.encode_as_ids(sentence)
        decoded = sp.decode_pieces(tokens)
        print(f"\nInput: {sentence}")
        print(f"Tokens: {tokens}")
        print(f"IDs: {ids}")
        print(f"Decoded: {decoded}")

    print(f"\nVocabulary size: {sp.vocab_size()}")
    print(f"\nSpecial tokens and their IDs:")
    for token in special_tokens:
        token_id = sp.piece_to_id(token)
        print(f"  {token:20s} -> ID {token_id}")


if __name__ == "__main__":
    main()
