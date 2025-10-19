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
        Get a random batch of padded data for training, validation or test.
        
        In train / validation mode, pad the sequences so they are all of size context_size
        and load batch_size.

        In test mode, sequences are trimmed to context_size, but not padded, load batch_size 1
        because tensors of different shape (no padding).
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
        input_ids_w_batch, labels_w_batch = [], []
        input_ids_l_batch, labels_l_batch = [], []

        for i in ix:
            example = dataset[i.item()]
            # input_ids and labels are lists of two lists: [accepted, rejected]
            input_ids = example['input_ids']
            labels = example['labels']
            rewards = example['rewards']

            if split in ('train', 'val'):
                for input_id, label, reward in zip(input_ids, labels, rewards):
                    
                    # Make a copy to avoid modifying the list in place
                    current_input_id = list(input_id)
                    current_label = list(label)

                    # Truncate if longer than context_size
                    if len(current_input_id) > self.context_size:
                        current_input_id = current_input_id[-self.context_size:]
                        current_label = current_label[-self.context_size:]
                    
                    # Pad if shorter than context_size
                    pad_length = self.context_size - len(current_input_id)
                    if (pad_length > 0) & (split != 'test'):
                        current_input_id.extend([self.pad_token_id] * pad_length)
                        current_label.extend([-100] * pad_length) # -100 is the ignore_index for loss calculation
                    
                    if reward > 0: # Accepted response
                        input_ids_w_batch.append(torch.tensor(current_input_id, dtype=torch.long))
                        labels_w_batch.append(torch.tensor(current_label, dtype=torch.long))
                    else: # Rejected response
                        input_ids_l_batch.append(torch.tensor(current_input_id, dtype=torch.long))
                        labels_l_batch.append(torch.tensor(current_label, dtype=torch.long))
            elif split == 'test':
                # For testing, we return the prompt and the accepted completion.
                # This allows for both generation from the prompt and scoring of the completions.
                
                # The raw data has one prompt and two completions (accepted, rejected).
                # The tokenized data has two sequences: [prompt+accepted] and [prompt+rejected].
                # We need to extract the prompt and both completions.
                
                accepted_input_ids = input_ids[0]
                accepted_labels = labels[0]

                # Find where the prompt ends in the accepted sequence.
                # The prompt part has labels of -100.
                try:
                    # The prompt is the input sequence up to the first non-masked label.
                    prompt_end_idx = accepted_labels.index(next(l for l in accepted_labels if l != -100))
                except StopIteration: # Should not happen in RL dataset
                    prompt_end_idx = len(accepted_input_ids)
                
                prompt = torch.tensor(accepted_input_ids[:prompt_end_idx], dtype=torch.long).view(1, -1)
                accepted_completion = torch.tensor([l for l in accepted_labels if l != -100], dtype=torch.long).view(1, -1)                
                # For test, we return the tensors directly without batching/stacking
                return prompt.to(self.device), accepted_completion.to(self.device)

        x_w, y_w = torch.stack(input_ids_w_batch), torch.stack(labels_w_batch)
        x_l, y_l = torch.stack(input_ids_l_batch), torch.stack(labels_l_batch)

        # Move tensors to the correct device
        if self.device == 'cuda':
            return (
                (x_w.pin_memory().to(self.device, non_blocking=True), 
                 y_w.pin_memory().to(self.device, non_blocking=True)),
                (x_l.pin_memory().to(self.device, non_blocking=True),
                 y_l.pin_memory().to(self.device, non_blocking=True))
            )
        else:
            return (
                (x_w.to(self.device), y_w.to(self.device)),
                (x_l.to(self.device), y_l.to(self.device))
            )
