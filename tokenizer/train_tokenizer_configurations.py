"""
Configuration definitions for tokenizer training.

Each configuration specifies which datasets to use and how many shards to load from each.
The key is the configuration name, and the value is a dict mapping dataset names to shard counts.

Dataset names correspond to the directory names in the pre_training folder:
- institutional-books-1.0 (125 shards)
- tinystories (8 shards)
- wikitext-103-raw-v1 (8 shards)

Examples:
    - 'small': Use a small subset for quick testing
    - 'medium': Use a medium-sized corpus for development
    - 'full': Use all available data for production training
"""

DATASET_CONFIGURATIONS = {
    # Small configuration for quick testing
    'small': {
        'tinystories': 8,
        'wikitext-103-raw-v1': 8,
    },

    # Medium configuration for development
    'medium': {
        'institutional-books-1.0': 1,
        'tinystories': 1,
        'wikitext-103-raw-v1': 1,
    },

    # Full configuration for production training
    'full': {
        'institutional-books-1.0': 125,
        'tinystories': 8,
        'wikitext-103-raw-v1': 8,
    },

    # Books-focused configuration
    'books-only': {
        'institutional-books-1.0': 125,
    },

    # Test with single dataset
    'tinystories-only': {
        'tinystories': 8,
    },
}
