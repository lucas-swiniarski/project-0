# Simple Transformer

A simple Transformer library built with PyTorch.

## Setup

This project uses PyTorch. It's recommended to run it within a Python virtual environment.

1.  **Create and activate a conda environment:**
    ```bash
    conda create --name project-0-env python=3.10
    conda activate project-0-env
    ```

2.  **Install dependencies:**
    This will install PyTorch with CUDA support, Hugging Face Datasets, and Tokenizers.
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
    conda install -c conda-forge datasets tokenizers tensorboard
    ```

3. **Start tensorboard:**
    Change logdir to your model ckpt_dir/tensorboard
    ```bash
    screen
    tensorboard --logdir /home/lucas/project-0/checkpoints/25_10_15_pt_sft1/tensorboard/ --port 6006
    ```

## Pre-training

1.  **Download the dataset:**
    This script will download the TinyStories and WikiText datasets, process them, and save them to the specified directory.
    ```bash
    python3 -m dataset.pre_training.download_dataset --output-dir=/home/lucas/data/v1/raw/pre_training
    ```

2.  **Train the tokenizer:**
    This script trains a new BPE tokenizer on the training data created in the previous step and saves it.
    ```bash
    python3 -m tokenizer.train --dataset-dir /home/lucas/data/v1/raw/train --output-path /home/lucas/tokenizer/v1/tokenizer.json
    ```

3.  **Tokenize the datasets:**
    This script uses the trained tokenizer to convert the raw text datasets (train, validation, test) into sequences of token IDs.
    You can adjust the number of processes with the `--num-proc` flag.
    ```bash
    python3 -m tokenizer.tokenize_dataset \
        --dataset-dir /home/lucas/data/v1/raw/pre_training \
        --tokenizer-path /home/lucas/tokenizer/v1/tokenizer.json \
        --output-dir /home/lucas/data/v1/tokenized/pre_training/v2 \
        --num-proc 4
    ```

4.  **Inspect the tokenized data (Optional):**
    This script allows you to interactively view how the raw text from the test set was tokenized. It's useful for verifying that the tokenizer is working as expected.
    ```bash
    python3 -m tokenizer.inspect_tokenized_data \
        --raw-dataset-dir /home/lucas/data/v1/pre_training/raw/test \
        --tokenized-dataset-dir /home/lucas/data/v1/tokenized/v2/test \
        --tokenizer-path /home/lucas/tokenizer/v1/tokenizer.json
    ```

    This script counts the total number of tokens in the train, validation, and test sets after tokenization.
    ```bash
    python3 -m tokenizer.count_tokenized_data \
        --tokenized-dir /home/lucas/data/v1/tokenized/pre_training/v2
    ```    

5.  **Run the main script (example):**
    ```bash
    python3 pre_training.py
    ```

## Post-training - SFT

1.  **Download the dataset:**
    This script will download the Alpaca cleaned datasets, process them, and save them to the specified directory.
    ```bash
    python3 -m dataset.post_training.sft.download_dataset --output-dir=/home/lucas/data/v1/raw/post_training/sft --tokenizer-profile=post_training_v1
    ```

2.  **Tokenize the datasets:**
    This script uses the trained tokenizer to convert the raw text datasets (train, validation, test) into sequences of token IDs.
    You can adjust the number of processes with the `--num-proc` flag.
    ```bash
    python3 -m tokenizer.tokenize_dataset \
        --dataset-dir /home/lucas/data/v1/raw/post_training/sft \
        --tokenizer-path /home/lucas/tokenizer/v1/tokenizer.json \
        --output-dir /home/lucas/data/v1/tokenized/post_training/sft \
        --tokenizer-profile=post_training_v1 \
        --tokenizer-mode=post_training_sft \
        --num-proc 4
    ```

3.  **Inspect the tokenized data (Optional):**
    ```bash
    python3 -m tokenizer.inspect_tokenized_data \
        --raw-dataset-dir /home/lucas/data/v1/raw/post_training/sft/train \
        --tokenized-dataset-dir /home/lucas/data/v1/tokenized/post_training/sft/train \
        --tokenizer-path /home/lucas/tokenizer/v1/tokenizer.json
    ```

     This script counts the total number of tokens in the train, validation, and test sets after tokenization.
    ```bash
    python3 -m tokenizer.count_tokenized_data \
        --tokenized-dir /home/lucas/data/v1/tokenized/post_training/sft
    ```    

4. **Train model:**
    First change parameters of post_training_sft.py

    Then run
    ```bash
    python3 post_training_sft.py
    ```
    

## Post-training - DPO

## Orders of magnitude

1. Training capacity: 1 nvidia L4 - 24 GB vram - 2.42×10^14 FLOPS * 3600 * 24 = 2x10^19 FLOPs (f16)

2. Assuming FLOPs ~ 6 x N x D - N trainset tokens, D model params, 100 tokens per params, that's 170M params, 17B tokens.

3. 170M params x 2 bytes / param (f16) = 0.32 GB vram.

* TODO: architecture math. Guides: gpt-2 small is 12 layers, 768 hidden dim, need attention architecture + intermediate size / experts + embedding table.

4. vram: 0.32 gb for model + 1x for gradients, +2x for adam = 1.2 gb vram.
