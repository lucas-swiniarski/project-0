from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Metaspace as MetaspacePreTokenizer
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.processors import TemplateProcessing

def load_tokenizer(tokenizer_path: str) -> Tokenizer:
    """
    Loads a tokenizer from a file and configures its pre-tokenizer, decoder,
    and post-processor for consistent behavior across the project.

    Args:
        tokenizer_path (str): The path to the tokenizer.json file.

    Returns:
        Tokenizer: The configured tokenizer instance.
    """
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Set the pre-tokenizer to Metaspace to correctly handle spaces during decoding.
    # This is important for BPE models trained on text with spaces represented by ' '.
    tokenizer.pre_tokenizer = MetaspacePreTokenizer()

    # The decoder also needs to be explicitly set to Metaspace to correctly join
    # subword tokens and handle spaces.
    tokenizer.decoder = MetaspaceDecoder()

    # Set up the post-processor to correctly handle special tokens during decoding.
    tokenizer.post_processor = TemplateProcessing(
        single="$A",  # The main sequence.
        special_tokens=[
            ("[SOS]", tokenizer.token_to_id("[SOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )
    return tokenizer
