from abc import ABC, abstractmethod
from typing import Any, Dict, List

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.models import BPE, Unigram
from tokenizers.pre_tokenizers import Metaspace as MetaspacePreTokenizer
from tokenizers.processors import ByteLevel, TemplateProcessing


class TokenizerProfile(ABC):
    """
    Abstract Base Class for a tokenizer profile.
    A profile defines a set of special tokens and the logic for using them
    in formatting and tokenization for a specific training phase.
    """

    @abstractmethod
    def get_special_tokens(self) -> List[str]:
        """
        Returns the list of special tokens.
        """
        pass
    
    @abstractmethod
    def get_pad_token(self) -> str:
        """
        Returns the pad token.
        """
        pass
    
    @abstractmethod
    def get_stop_token(self) -> str:
        """
        Returns the stop token.
        """
        pass
    
    @abstractmethod
    def create_tokenizer(self) -> Tokenizer:
        """
        Init a tokenizer.
        """
        pass

    @abstractmethod
    def get_trainer(self, vocab_size: int):
        """
        Returns the trainer for this tokenizer profile.
        """
        pass

    @abstractmethod
    def format_example(self, example: Dict[str, Any], mode: str | None) -> Dict[str, Any]:
        """
        Takes a raw data record (e.g., from Hugging Face) and formats it
        into the structure required for tokenization.
        """
        pass
    
    @abstractmethod
    def tokenized_columns_to_remove(self, mode: str | None) -> List[str]:
        """
        Returns the list of columns to remove when tokenizing a dataset.
        """ 
        pass

    @abstractmethod
    def tokenize_datasets(self, 
                          examples: Dict[str, List], 
                          tokenizer: Tokenizer, 
                          mode: str | None = None) -> Dict[str, List[Any]]:
        """
        Takes a batch of formatted examples and tokenizes them.
        This is where logic like loss masking for SFT would live.
        """
        pass
    
    @abstractmethod
    def configure_tokenizer(self, tokenizer: Tokenizer) -> Tokenizer:
        """
        Loads a tokenizer from a file.
        """
        pass


class PreTrainingV1(TokenizerProfile):
    """Profile for the pre-training phase."""
    
    def get_special_tokens(self) -> List[str]:
        return ["[UNK]", "[PAD]", "[SOS]", "[EOS]"]
    
    def get_pad_token(self):
        return "[PAD]"
    
    def get_stop_token(self):
        return "[EOS]"
    
    def create_tokenizer(self) -> Tokenizer:
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        return tokenizer

    def get_trainer(self, vocab_size: int):
        from tokenizers.trainers import BpeTrainer
        return BpeTrainer(vocab_size=vocab_size, special_tokens=self.get_special_tokens())

    def format_example(self, example: Dict[str, Any], mode: str | None) -> Dict[str, Any]:
        return {"text": example["text"]}
    
    def tokenized_columns_to_remove(self, mode: str | None) -> List[str]:
        return ["text"]

    def tokenize_datasets(self, 
                          examples: Dict[str, List], 
                          tokenizer: Tokenizer, 
                          mode: str | None = None) -> Dict[str, List[Any]]:
        """Tokenizes text and adds SOS/EOS tokens."""
        output = tokenizer.encode_batch(examples["text"])
        sos_token_id = tokenizer.token_to_id("[SOS]")
        eos_token_id = tokenizer.token_to_id("[EOS]")

        all_token_ids = []
        for encoding in output:
            all_token_ids.append([sos_token_id] + encoding.ids + [eos_token_id])

        return {"input_ids": all_token_ids}
    
    def configure_tokenizer(self, tokenizer: Tokenizer) -> Tokenizer:
        """
        Configures a loaded tokenizer with the correct pre-tokenizer, decoder,
        and post-processor for pre-training.
        """
        # Set the pre-tokenizer to Metaspace to correctly handle spaces during decoding.
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


class PostTrainingV1(TokenizerProfile):
    """Profile for Supervised Fine-Tuning with the Alpaca format.
    It must support PreTrainingV1.
    """

    def get_special_tokens(self) -> List[str]:
        return PreTrainingV1().get_special_tokens() + ["<SYSTEM>", "</SYSTEM>", "<USER>", "</USER>", "<MODEL>", "</MODEL>"]
    
    def get_pad_token(self):
        return "[PAD]"
    
    def get_stop_token(self):
        return "</MODEL>"
    
    def create_tokenizer(self) -> Tokenizer:
        # Backward compatibility.
        return PreTrainingV1().create_tokenizer()  

    def get_trainer(self, vocab_size: int):
        # This profile is intended for fine-tuning, but we can allow training a tokenizer.
        return PreTrainingV1().get_trainer(vocab_size)

    def format_example(self, example: Dict[str, Any], mode: str | None) -> Dict[str, Any]:
        """Formats an example based on the specified mode."""
        if mode == 'pre_training':
            # Backward compatibility.
            return PreTrainingV1().format_example(example, mode)
        elif mode == 'alpaca':
            system_prompt = f"<SYSTEM>{example['instruction']}</SYSTEM>"
            user_prompt = f"<USER>{example['input']}</USER>" if example.get('input') else ""
            prompt = f"{system_prompt}{user_prompt}<MODEL>"
            completion = f"{example['output']}</MODEL>"
            return {"prompt": prompt, "completion": completion}
        elif mode == 'openassistant':
            prompt = ''
            for context in example['context']:
                if context['role'] == 'prompter':
                    prompt += f"<USER>{context['text']}</USER>"
                elif context['role'] == 'assistant':
                    prompt += f"<MODEL>{context['text']}</MODEL>"
                else:
                    raise ValueError(f"Unknown role: {context['role']}")
            prompt += '<MODEL>'
            accepted_completion = f"{example['accepted']['text']}</MODEL>"
            rejected_completion = f"{example['rejected']['text']}</MODEL>"
            return {"prompt": prompt, "accepted": accepted_completion, "rejected": rejected_completion}
            

        raise ValueError(f"Unknown mode: {mode}")
    
    def tokenized_columns_to_remove(self, mode: str | None) -> List[str]:
        if mode == 'pre_training':
            # Backward compatibility.
            return PreTrainingV1().tokenized_columns_to_remove(mode)
        elif mode == 'post_training_sft':
            return ["prompt", "completion"]
        elif mode == 'post_training_rl':
            return ["prompts", "accepted", "rejected"]
        raise ValueError(f'Mode: {mode} not supported.')

    def tokenize_datasets(self, 
                          examples: Dict[str, List], 
                          tokenizer: Tokenizer, 
                          mode: str | None = None) -> Dict[str, List[Any]]:
        """Tokenizes the prompt (x1) and completion (y1) together."""
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided for SFT mode.")

        if mode == 'pre_training':
            # Backward compatibility.
            return PreTrainingV1().tokenize_datasets(examples, tokenizer, mode)
        elif mode == 'post_training_sft':
            prompts = examples['prompt']
            completions = examples['completion']

            # Tokenize prompts and completions separately to get their lengths
            tokenized_prompts = tokenizer.encode_batch(prompts)
            tokenized_completions = tokenizer.encode_batch(completions)

            input_ids = []
            labels = []

            for prompt_encoding, completion_encoding in zip(tokenized_prompts, tokenized_completions):
                prompt_ids = prompt_encoding.ids
                completion_ids = completion_encoding.ids

                # Create the full sequence and the corresponding labels with masking for the prompt
                full_input_ids = prompt_ids + completion_ids
                full_labels = ([-100] * len(prompt_ids)) + completion_ids

                # The model predicts the next token. Therefore, the input `x` should be the sequence
                # up to the second-to-last token, and the target `y` (our labels) should be the
                # sequence shifted by one to the left (from the second token to the end).
                # We do this shift here so the data loader can be simpler.
                input_ids.append(full_input_ids[:-1])
                labels.append(full_labels[1:])

            return {"input_ids": input_ids, "labels": labels}
        elif mode == 'post_training_rl':
            # examples: {'prompts': prompts, 'accepted': accepted_responses, 'rejected': rejected_responses}
            prompts = examples['prompts']
            accepted_responses = examples['accepted']
            rejected_responses = examples['rejected']
            
            tokenized_prompts = tokenizer.encode_batch(prompts)
            tokenized_accepted_responses = tokenizer.encode_batch(accepted_responses)
            tokenized_rejected_responses = tokenizer.encode_batch(rejected_responses)
            
            input_ids = []
            labels = []
            rewards = []
            
            for prompts_encoding, accepted_encoding, rejected_encoding in zip(
                tokenized_prompts, 
                tokenized_accepted_responses, 
                tokenized_rejected_responses):
                prompt_ids = prompts_encoding.ids
                accepted_ids = accepted_encoding.ids
                rejected_ids = rejected_encoding.ids

                full_accepted_ids = prompt_ids + accepted_ids
                full_rejected_ids = prompt_ids + rejected_ids
                
                full_accepted_labels = [-100] * len(prompt_ids) + accepted_ids
                full_rejected_labels = [-100] * len(prompt_ids) + rejected_ids
                
                input_ids.append([full_accepted_ids[:-1], full_rejected_ids[:-1]])
                labels.append([full_accepted_labels[1:], full_rejected_labels[1:]])
                rewards.append([1.0, -1.0])
            
            return {"input_ids": input_ids, "labels": labels, "rewards": rewards}

        raise ValueError(f"Unknown mode: {mode}")

    def configure_tokenizer(self, tokenizer: Tokenizer) -> Tokenizer:
        original_vocab_size = tokenizer.get_vocab_size()        
        new_special_tokens = []
        for special_token in self.get_special_tokens():
            if special_token not in tokenizer.get_vocab():
                new_special_tokens += [special_token]
        tokenizer.add_special_tokens(new_special_tokens)
        new_vocab_size = tokenizer.get_vocab_size()

        # Set the pre-tokenizer to Metaspace to correctly handle spaces during decoding.
        # This is important for BPE models trained on text wi
        # th spaces represented by ' '.
        tokenizer.pre_tokenizer = MetaspacePreTokenizer()
        tokenizer.decoder = MetaspaceDecoder()

        # Set up the post-processor to correctly handle special tokens during decoding.
        tokenizer.post_processor = TemplateProcessing(
            single="$A",  # The main sequence.
            special_tokens=[
                (special_token, tokenizer.token_to_id(special_token))
                for special_token in self.get_special_tokens()
            ],
        )
        return tokenizer


class SentencePieceV2(TokenizerProfile):
    """Profile for the pre-training phase."""
    
    def get_special_tokens(self) -> List[str]:
        return ["[UNK]", "[PAD]", "[SOS]", "[EOS]", "<SYSTEM>", "</SYSTEM>", "<USER>", "</USER>", "<MODEL>", "</MODEL>"]
    
    def get_pad_token(self):
        return "[PAD]"
    
    def get_stop_token(self):
        return "</MODEL>"
    
    def create_tokenizer(self) -> Tokenizer:
        tokenizer = Tokenizer(Unigram())
        return tokenizer

    def get_trainer(self, vocab_size: int):
        from tokenizers.trainers import UnigramTrainer
        return UnigramTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            special_tokens=self.get_special_tokens(),
            unk_token="[UNK]"
        )

    def format_example(self, example: Dict[str, Any], mode: str | None) -> Dict[str, Any]:
        return {"text": example["text"]}
    
    def tokenized_columns_to_remove(self, mode: str | None) -> List[str]:
        return ["text"]

    def tokenize_datasets(self, 
                          examples: Dict[str, List], 
                          tokenizer: Tokenizer, 
                          mode: str | None = None) -> Dict[str, List[Any]]:
        """Tokenizes text and adds SOS/EOS tokens."""
        if mode == 'pre_training':
            output = tokenizer.encode_batch(examples["text"])
            sos_token_id = tokenizer.token_to_id("[SOS]")
            eos_token_id = tokenizer.token_to_id("[EOS]")

            all_token_ids = []
            for encoding in output:
                all_token_ids.append([sos_token_id] + encoding.ids + [eos_token_id])

            return {"input_ids": all_token_ids}
        raise NotImplementedError(f'Mode: {mode} not implemented.')
    
    def configure_tokenizer(self, tokenizer: Tokenizer) -> Tokenizer:
        """
        Configures a loaded tokenizer with the correct pre-tokenizer, decoder,
        and post-processor for pre-training.
        """
        # Set the pre-tokenizer to Metaspace to correctly handle spaces during decoding.
        tokenizer.pre_tokenizer = MetaspacePreTokenizer()

        # The decoder also needs to be explicitly set to Metaspace to correctly join
        # subword tokens and handle spaces.
        tokenizer.decoder = MetaspaceDecoder()

        # Set up the post-processor to correctly handle special tokens during decoding.
        tokenizer.post_processor = TemplateProcessing(
            single="$A",  # The main sequence.
            special_tokens=[
                (token, tokenizer.token_to_id(token)) for token in self.get_special_tokens()
            ],
        )
        return tokenizer

TOKENIZER_NAME_TO_PROFILE = {
    'pre_training_v1': PreTrainingV1,
    'post_training_v1': PostTrainingV1,
    'sentence_piece_v2': SentencePieceV2,
}