# QLoRA SFT Template (Qwen-style Models)

A clean, open-source template for supervised fine-tuning with QLoRA.
This repository includes data preparation, training, and vLLM serving scripts.

## What is included

- `prepare_sft_dataset.py`: convert raw text records into Alpaca-style JSONL.
- `train_qlora_sft.py`: direct Trainer-based QLoRA SFT script.
- `configs/qlora.yaml`: LLaMA-Factory training config template.
- `configs/ds_zero2.json`: DeepSpeed ZeRO-2 config.
- `scripts/train_deepspeed.sh`: LLaMA-Factory + DeepSpeed launch helper.
- `scripts/serve_vllm_lora.sh`: vLLM API server with LoRA adapter.

## What is not included

Original project training data is removed for privacy/compliance.
See `DATA.md` and `data/README.md` for expected schema.

## Quick start

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Prepare your data

Create files:

- `data/train.jsonl`
- Optional: `data/val.jsonl`

Each line must follow:

```json
{"instruction": "...", "input": "...", "output": "..."}
```

### 3) Configure paths

Edit `configs/qlora.yaml`:

- `model_name_or_path`
- `dataset_dir`
- `output_dir`

Or use shell variables from `.env.example`.

### 4) Train with LLaMA-Factory + DeepSpeed

```bash
bash scripts/train_deepspeed.sh /abs/path/to/base-model data
```

### 5) Train with standalone script (optional)

```bash
python train_qlora_sft.py \
  --model_name_or_path /abs/path/to/base-model \
  --train_file data/train.jsonl \
  --eval_file data/val.jsonl \
  --output_dir outputs/qwen3-30b-a3b-lora-material
```

### 6) Serve LoRA adapter with vLLM

```bash
bash scripts/serve_vllm_lora.sh \
  /abs/path/to/base-model \
  outputs/qwen3-30b-a3b-lora-material \
  8000
```

## Security checklist before push

- Ensure `data/*.jsonl` are not committed.
- Ensure `.env` is not committed.
- Ensure no internal paths, hostnames, tokens, or passwords remain.

## License

MIT. See `LICENSE`.
