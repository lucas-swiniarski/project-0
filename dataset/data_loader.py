import os
import torch
import numpy as np
from datasets import load_from_disk

class DataLoader:
    """
    Loads and prepares tokenized data for training.
    It memory-maps the tokenized data, treating it as one large contiguous sequence.
    """
    def __init__(self, tokenized_dir, context_size, batch_size, device='cpu'):
        self.context_size = context_size
        self.batch_size = batch_size
        self.device = device
        self.device_type = device_type

        # Load datasets for train and validation
        train_data_path = os.path.join(tokenized_dir, 'train')
        val_data_path = os.path.join(tokenized_dir, 'validation')

        self.train_tokens = self._load_and_mmap(train_data_path, "train_tokens.bin")
        self.val_tokens = self._load_and_mmap(val_data_path, "val_tokens.bin")

        print(f"Loaded training data with {len(self.train_tokens):,} tokens.")
        print(f"Loaded validation data with {len(self.val_tokens):,} tokens.")

    def _load_and_mmap(self, dataset_path, mmap_filename):
        """
        Loads a dataset, concatenates all 'input_ids' into a single array,
        saves it to a binary file, and then memory-maps it.
        This avoids loading the entire dataset into RAM.
        """
        # Check if the memory-mapped file already exists
        if os.path.exists(mmap_filename):
            print(f"Found existing memory-mapped file: {mmap_filename}. Loading it.")
            # The data type is uint16 because vocab size is < 65535
            tokens = np.memmap(mmap_filename, dtype=np.uint16, mode='r')
            return tokens

        print(f"Memory-mapped file not found. Creating it from {dataset_path}...")
        try:
            dataset = load_from_disk(dataset_path)
        except FileNotFoundError:
            print(f"Error: Dataset not found at {dataset_path}")
            print("Please run the tokenization scripts first.")
            exit(1)

        # Concatenate all token sequences into one large numpy array
        # Using uint16 is memory-efficient for vocab_size < 65535
        total_tokens = sum(len(x) for x in dataset['input_ids'])
        arr = np.memmap(mmap_filename, dtype=np.uint16, mode='w+', shape=(total_tokens,))
        
        print(f"Concatenating {len(dataset)} documents into a single stream...")
        idx = 0
        for example in dataset:
            tokens = example['input_ids']
            arr[idx : idx + len(tokens)] = tokens
            idx += len(tokens)
        
        arr.flush()
        return arr
    
    def get_n_batches_per_epoch(self):
        return len(self.train_tokens) // (self.batch_size * self.context_size)
    
    def get_batch(self, split):
        """
        Get a random batch of data for training or validation.
        """
        data = self.train_tokens if split == 'train' else self.val_tokens
        
        # Generate random starting indices for each sequence in the batch
        ix = torch.randint(len(data) - self.context_size-1, (self.batch_size,))
        
        # Create input sequences (x) and target sequences (y)
        x = torch.stack([torch.from_numpy(data[i:i+self.context_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i+1:i+1+self.context_size].astype(np.int64)) for i in ix])

        # Move tensors to the correct device
        if self.device == 'cuda':
            return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            return x.to(self.device), y.to(self.device)
