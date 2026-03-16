# Data Directory

This directory intentionally does not include original training data.

## Required training files

Create the following files with your own data:

- `train.jsonl` (required)
- `val.jsonl` (optional)

## Record schema

Each line is one JSON object:

```json
{"instruction": "task", "input": "context", "output": "answer"}
```

## Notes

- Keep UTF-8 encoding.
- One JSON object per line.
- Avoid multiline raw newlines inside a single JSON value unless escaped.
- `dataset_info.json` already maps these keys for LLaMA-Factory.
