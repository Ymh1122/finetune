# Data Statement

This repository is published as a generic LoRA fine-tuning template.

Original project data is not included in this public release.

## Why data is removed

The original dataset contains private domain records and cannot be redistributed.
To prevent accidental disclosure, all JSONL training files were removed from the repository.

## Expected data format

Use Alpaca-style JSONL records with three fields:

```json
{"instruction": "...", "input": "...", "output": "..."}
```

Each line must be a valid JSON object.

## Required files

- `data/train.jsonl`
- Optional: `data/val.jsonl`

`data/dataset_info.json` already maps these fields for LLaMA-Factory.

## Build your own dataset

You can adapt your raw source data with:

```bash
python prepare_sft_dataset.py --help
```
