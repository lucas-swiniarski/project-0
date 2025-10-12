import os
import torch
from datasets import load_from_disk

class DataLoader:
    """
    Loads and prepares tokenized data for Post-Training Supervised Fine-Tuning (SFT).
    It samples individual examples from the dataset and pads them to a fixed context size.
    """
    def __init__(self, 
                 tokenized_dir: str, 
                 context_size: int, 
                 batch_size: int, 
                 pad_token_id: int, 
                 device='cpu'):
        self.context_size = context_size
        self.batch_size = batch_size
        self.device = device
        self.pad_token_id = pad_token_id

        # Load datasets for train and validation
        train_data_path = os.path.join(tokenized_dir, 'train')
        val_data_path = os.path.join(tokenized_dir, 'validation')
        test_data_path = os.path.join(tokenized_dir, 'test')

        try:
            self.train_dataset = load_from_disk(train_data_path)
            self.val_dataset = load_from_disk(val_data_path)
            self.test_dataset = load_from_disk(test_data_path)
        except FileNotFoundError as e:
            print(f"Error: Dataset not found at {train_data_path} or {val_data_path}")
            print(f"Details: {e}")
            print("Please run the tokenization scripts for SFT first.")
            exit(1)

        print(f"Loaded training data with {len(self.train_dataset):,} examples.")
        print(f"Loaded validation data with {len(self.val_dataset):,} examples.")

    
    def get_n_batches_per_epoch(self):
        return len(self.train_dataset) // self.batch_size
    
    def get_batch(self, split):
        """
        Get a random batch of padded data for training or validation.
        
        In train / validation mode, pad the sequences so they are all of size context_size
        and load batch_size.

        In test mode, sequences are trimmed to context_size, but not padded, load batch_size 1
        because tensors of differnet shape (no padding).
        """
        data = ''
        if split in ('train', 'val'):
            dataset = self.train_dataset if split == 'train' else self.val_dataset
            ix = torch.randint(len(dataset), (self.batch_size,))
        elif split == 'test':
            dataset = self.test_dataset
            ix = torch.randint(len(dataset), (1,))
        else:
            raise ValueError(f"Invalid split: {split}")
              
        # Retrieve and pad each example
        input_ids_batch = []
        labels_batch = []

        for i in ix:
            example = dataset[i.item()]
            input_ids = example['input_ids']
            labels = example['labels']
            
            
            if split in ('train', 'val'):
                # Truncate if longer than context_size
                if len(input_ids) > self.context_size:
                    input_ids = input_ids[-self.context_size:]
                    labels = labels[-self.context_size:]
                
                # Pad if shorter than context_size
                pad_length = self.context_size - len(input_ids)
                if (pad_length > 0) & (split != 'test'):
                    input_ids.extend([self.pad_token_id] * pad_length)
                    labels.extend([-100] * pad_length) # -100 is the ignore_index for loss calculation
                input_ids_batch.append(torch.tensor(input_ids, dtype=torch.long))
                labels_batch.append(torch.tensor(labels, dtype=torch.long))
            elif split == 'test':
                # For testing, we want to evaluate the model's generation.
                # The input to the model should be the prompt only.
                # The labels are the expected completion.

                # Find where the prompt ends (where labels are not -100)
                try:
                    first_label_idx = labels.index(next(l for l in labels if l != -100)) + 1
                except StopIteration: # No labels found, use entire sequence as input
                    first_label_idx = len(labels)

                # The prompt is everything up to the first real label.
                prompt_ids = input_ids[:first_label_idx]
                input_ids_batch.append(torch.tensor(prompt_ids, dtype=torch.long))
                labels_batch.append(torch.tensor(labels[first_label_idx:], dtype=torch.long))

        # Stack into tensors
        x = torch.stack(input_ids_batch)
        y = torch.stack(labels_batch)
        # Move tensors to the correct device
        if self.device == 'cuda':
            return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            return x.to(self.device), y.to(self.device)
