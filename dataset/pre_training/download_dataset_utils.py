from datasets import Dataset
import os


def write_datasets(split_to_dataset: dict[str, Dataset], 
                   output_dir: str,
                   dataset_name: str):
    """Write all splits dataset under output_dir/dataset_name/{split_name}/
    
    Args:
        split_to_dataset (dict[str, Datset]) a dict from split name 
            (train) to a Dataset object to write.
        output_dir (str): The root directory to save the dataset to.
        dataset_name (str): Name of the dataset all attached. 
    """
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    for split, dataset in split_to_dataset.items():
        dataset.save_to_disk(os.path.join(dataset_dir, split))