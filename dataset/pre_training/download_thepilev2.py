
import os
import argparse
from datasets import load_dataset, concatenate_datasets

def main():
    ds = load_dataset("bigcode/the-stack-v2", streaming=True, split="train")
    i = 0
    for sample in iter(ds): 
        print(sample) 
        i += 1
        if i > 10:
            break

if __name__ == "__main__":
    main()