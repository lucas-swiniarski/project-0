import os
import argparse
import random
from datasets import load_dataset, Dataset

import tokenizer.profiles as tokenizer_profiles
from . import download_dataset_utils as utils

def main():
    parser = argparse.ArgumentParser(description="Load, process, and save the OpenAssistant dataset for RL.")
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/home/lucas/data/v1/raw/post_training/rl',
        help='The directory to save the processed datasets.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility of dataset splits.'
    )
    parser.add_argument(
        '--tokenizer-profile',
        type=str,
        default='post_training_v1',
        help='Random seed for reproducibility of dataset splits.'
    )
    parser.add_argument(
        '--num-test-roots',
        type=int,
        default=500,
        help='Number of conversation roots to use for the test set.
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    tokenizer_profile = tokenizer_profiles.TOKENIZER_NAME_TO_PROFILE[args.tokenizer_profile]()
    os.makedirs(output_dir, exist_ok=True)

    # --- Load Dataset ---
    print("Loading OpenAssistant/oasst1 dataset...")
    open_assistant_dataset = load_dataset("OpenAssistant/oasst1")
    original_train_split = open_assistant_dataset['train']
    validation_split = open_assistant_dataset['validation']

    # --- Split Dataset ---
    print("Splitting train set into train and test sets by conversation root...")
    # Build graph structures to identify conversation roots
    _, parent_id_to_message_ids, root_ids = utils.build_graph_structures(original_train_split)
    
    # Shuffle roots for random sampling
    random.seed(args.seed)
    random.shuffle(root_ids)
    
    # Sample roots for the test set
    test_root_ids = root_ids[:args.num_test_roots]
    
    # Collect all message IDs belonging to the test conversation threads
    test_message_ids = set()
    for root_id in test_root_ids:
        test_message_ids.update(utils.get_thread_messages(root_id, parent_id_to_message_ids))
        
    # Create the new splits by filtering the original dataset
    train_split = original_train_split.filter(lambda x: x['message_id'] not in test_message_ids)
    test_split = original_train_split.filter(lambda x: x['message_id'] in test_message_ids)

    # --- Process and Format Datasets ---
    def process_split(split, split_name):
        print(f"Processing '{split_name}' split...")
        message_id_to_message, parent_id_to_message_ids, root_ids = utils.build_graph_structures(split)
        print(f"Found {len(root_ids)} conversation roots in '{split_name}' split.")
        
        prompts = []
        accepted_responses = []
        rejected_responses = []
        
        for root_id in root_ids:
            for example in utils.traverse_thread(
                root_id, 
                message_id_to_message, 
                parent_id_to_message_ids, 
                tokenizer_profile):
                prompts += [example['prompt']]
                accepted_responses += [example['accepted']]
                rejected_responses += [example['rejected']]
        
        print(f"Found {len(prompts)} preferrence pairs in '{split_name}' split.")
        # Create a new Hugging Face Dataset from the processed examples
        # Assuming format_rl_example returns a dict with a 'text' key
        return Dataset.from_dict({'prompts': prompts, 'accepted': accepted_responses, 'rejected': rejected_responses})

    train_set = process_split(train_split, 'train')
    validation_set = process_split(validation_split, 'validation')
    test_set = process_split(test_split, 'test')

    # --- Save Datasets to Disk ---
    print(f"Saving processed datasets to {output_dir}...")
    train_set.save_to_disk(os.path.join(output_dir, 'train/openassistant'))
    validation_set.save_to_disk(os.path.join(output_dir, 'validation/openassistant'))
    test_set.save_to_disk(os.path.join(output_dir, 'test/openassistant'))

    print("Dataset processing and saving complete.")
    print(f"Train examples: {len(train_set)}, Validation examples: {len(validation_set)}, Test examples: {len(test_set)}")

if __name__ == "__main__":
    main()