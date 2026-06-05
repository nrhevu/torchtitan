import os
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

dataset_path = os.environ["DATASET_PATH"]
tokenizer_name = os.environ["TOKENIZER_NAME"]
output_path = Path(os.environ["OUTPUT_PATH"])
num_proc = int(os.environ["NUM_PROC"])
max_train_samples = int(os.environ.get("MAX_TRAIN_SAMPLES", "0"))

if output_path.exists() and any(output_path.iterdir()):
    raise FileExistsError(
        f"{output_path} already exists and is not empty. Move or remove it before rerunning."
    )

output_path.mkdir(parents=True, exist_ok=True)

dataset = load_dataset(dataset_path, split="train")
if max_train_samples > 0:
    sample_count = min(max_train_samples, len(dataset))
    print(f"Selecting first {sample_count} examples for this smoke run...")
    dataset = dataset.select(range(sample_count))

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

def tokenize_function(examples):
    encodings = tokenizer.backend_tokenizer.encode_batch_fast(examples["text"])
    return {
        "input_ids": [encoding.ids for encoding in encodings],
        "attention_mask": [encoding.attention_mask for encoding in encodings],
    }

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=num_proc)

print("Saving tokenized dataset...")
tokenized_dataset.save_to_disk(str(output_path))