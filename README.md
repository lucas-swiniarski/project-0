# Simple Transformer

A simple Transformer library built with PyTorch.

## Setup and Running

This project uses PyTorch. It's recommended to run it within a Python virtual environment.

1.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install dependencies:**
    This will install Hugging Face Datasets, and Tokenizers.
    ```bash
    pip install datasets tokenizers
    ```

3.  **Prepare the dataset:**
    This script will download the TinyStories and WikiText datasets, process them, and save them to the specified directory.
    ```bash
    python3 -m dataset.load_datasets --output-dir=/home/lucas/data/v1
    ```

4.  **Train the tokenizer:**
    This script trains a new BPE tokenizer on the training data created in the previous step and saves it.
    ```bash
    python3 -m tokenizer.train --dataset-dir /home/lucas/data/v1/train --output-path /home/lucas/tokenizer/v1/tokenizer.json
    ```

5.  **Run the main script (example):**
    ```bash
    python3 main.py
    ```
```