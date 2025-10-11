import os
import torch
from datasets import load_from_disk

class DataLoader:
    """
    Loads and prepares tokenized data for Post-Training Supervised Fine-Tuning (SFT).
    It samples individual examples from the dataset and pads them to a fixed context size.
    """
    def __init__(self, tokenized_dir, context_size, batch_size, pad_token_id, device='cpu'):
        self.context_size = context_size
        self.batch_size = batch_size
        self.device = device
        self.pad_token_id = pad_token_id

        # Load datasets for train and validation
        train_data_path = os.path.join(tokenized_dir, 'train')
        val_data_path = os.path.join(tokenized_dir, 'validation')

        try:
            self.train_dataset = load_from_disk(train_data_path)
            self.val_dataset = load_from_disk(val_data_path)
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
        """
        dataset = self.train_dataset if split == 'train' else self.val_dataset
        
        # Generate random indices for each example in the batch
        ix = torch.randint(len(dataset), (self.batch_size,))
        
        # Retrieve and pad each example
        input_ids_batch = []
        labels_batch = []

        for i in ix:
            example = dataset[i.item()]
            input_ids = example['input_ids']
            labels = example['labels']

            # Truncate if longer than context_size
            if len(input_ids) > self.context_size:
                input_ids = input_ids[-self.context_size:]
                labels = labels[-self.context_size:]
            
            # Pad if shorter than context_size
            pad_length = self.context_size - len(input_ids)
            if pad_length > 0:
                input_ids.extend([self.pad_token_id] * pad_length)
                labels.extend([-100] * pad_length) # -100 is the ignore_index for loss calculation
            input_ids_batch.append(torch.tensor(input_ids, dtype=torch.long))
            labels_batch.append(torch.tensor(labels, dtype=torch.long))

        # Stack into tensors
        x = torch.stack(input_ids_batch)
        y = torch.stack(labels_batch)
        # Move tensors to the correct device
        if self.device == 'cuda':
            return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            return x.to(self.device), y.to(self.device)
