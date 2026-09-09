# Qwen Domain Fine-Tuning & Serving Toolkit

### End-to-End QLoRA / SFT Training and vLLM Serving for Qwen-Style Models

> Sanitized engineering deliverable derived from my AI application internship at the Guangdong Academy of Sciences.  
> Original domain data, internal paths, credentials, and proprietary materials are not included in this public release.

This repository provides a compact, reproducible pipeline for **domain adaptation of Qwen-style large language models**, covering:

- instruction-data preparation;
- LoRA / QLoRA supervised fine-tuning;
- LLaMA-Factory configuration;
- DeepSpeed ZeRO-2 training;
- standalone Hugging Face / PEFT training;
- LoRA adapter deployment with vLLM.

The public version is intentionally generic so that the training and deployment workflow can be reused without exposing internship data or internal infrastructure.

---

## Project Overview

The repository represents the reusable engineering core of a domain-adaptation workflow developed during an AI application internship.

The original project involved domain-specific LLM adaptation and deployment.  
This public repository retains only the generalizable training and serving components:

```text
Raw Domain Records
        ↓
Data Cleaning / Normalization
        ↓
Instruction Dataset
        ↓
LoRA / QLoRA SFT
        ↓
LLaMA-Factory / DeepSpeed
        ↓
LoRA Adapter
        ↓
vLLM Serving
```

The goal is to make the **fine-tuning-to-serving path** easy to reproduce, modify, and deploy on a new domain dataset.

---

## What This Repository Includes

### 1. Instruction Data Preparation

`prepare_sft_dataset.py` converts raw records into an Alpaca-style instruction format:

```json
{
  "instruction": "...",
  "input": "...",
  "output": "..."
}
```

The resulting dataset can be used by both the standalone training script and the LLaMA-Factory configuration.

Public data files are intentionally excluded.

See:

- [`DATA.md`](DATA.md)
- [`data/README.md`](data/README.md)

---

### 2. QLoRA / LoRA Fine-Tuning

The repository provides two training paths.

#### Standalone Hugging Face / PEFT Training

```text
Transformers
    +
PEFT
    +
LoRA / QLoRA
    ↓
Adapter Checkpoint
```

Entry point:

```bash
python train_qlora_sft.py --help
```

This path is useful when direct control over the training loop and model configuration is preferred.

#### LLaMA-Factory + DeepSpeed Training

The repository also contains reusable configurations for:

- QLoRA fine-tuning;
- DeepSpeed ZeRO-2;
- dataset mapping;
- training hyperparameters.

Relevant files:

```text
configs/
├── qlora.yaml
└── ds_zero2.json
```

Launch helper:

```bash
scripts/train_deepspeed.sh
```

---

### 3. vLLM Adapter Serving

After training, the LoRA adapter can be served through vLLM without merging it permanently into the base model.

```text
Base Qwen Model
      +
LoRA Adapter
      ↓
vLLM Server
      ↓
OpenAI-Compatible API
```

Serving helper:

```bash
scripts/serve_vllm_lora.sh
```

This design keeps the base model reusable while allowing domain-specific adapters to be loaded independently.

---

## Repository Structure

```text
.
├── configs/
│   ├── qlora.yaml
│   └── ds_zero2.json
│
├── data/
│   ├── README.md
│   └── dataset_info.json
│
├── scripts/
│   ├── train_deepspeed.sh
│   └── serve_vllm_lora.sh
│
├── prepare_sft_dataset.py
├── train_qlora_sft.py
├── DATA.md
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Ymh1122/finetune.git
cd finetune
```

### 2. Create an Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Prepare Training Data

Create:

```text
data/train.jsonl
```

Optionally create:

```text
data/val.jsonl
```

Each line should follow the Alpaca-style schema:

```json
{"instruction": "...", "input": "...", "output": "..."}
```

For data preparation options:

```bash
python prepare_sft_dataset.py --help
```

See [`DATA.md`](DATA.md) for the public data boundary.

### 4. Configure the Model and Paths

Edit:

```text
configs/qlora.yaml
```

At minimum, verify:

```text
model_name_or_path
dataset_dir
output_dir
```

Environment-specific paths can also be provided through local configuration or variables based on `.env.example`.

### 5. Train with LLaMA-Factory + DeepSpeed

```bash
bash scripts/train_deepspeed.sh /abs/path/to/base-model data
```

The provided DeepSpeed configuration uses ZeRO-2 as the distributed-training template.

### 6. Train with the Standalone QLoRA Script

```bash
python train_qlora_sft.py \
  --model_name_or_path /abs/path/to/base-model \
  --train_file data/train.jsonl \
  --eval_file data/val.jsonl \
  --output_dir outputs/qwen-domain-lora
```

This route is optional if LLaMA-Factory is used.

### 7. Serve the LoRA Adapter with vLLM

```bash
bash scripts/serve_vllm_lora.sh \
  /abs/path/to/base-model \
  outputs/qwen-domain-lora \
  8000
```

The resulting service can be used as the inference endpoint for downstream applications.

---

## Engineering Design

The repository separates the workflow into four independent stages:

```text
1. Data Preparation
        ↓
2. Parameter-Efficient Fine-Tuning
        ↓
3. Adapter Artifact
        ↓
4. Inference Serving
```

This separation makes it possible to:

- rebuild datasets without changing training code;
- compare LoRA / QLoRA configurations;
- reuse a base model across multiple adapters;
- deploy a trained adapter independently;
- keep private domain data outside the public repository.

---

## Public Release and Data Boundary

This repository is a **sanitized public release**.

The original internship dataset is not redistributed because it contains private domain records.

The public repository therefore excludes:

- original domain training records;
- proprietary documents;
- internal service addresses;
- machine-specific paths;
- API keys and credentials;
- private checkpoints or outputs that should not be redistributed.

The expected public dataset format is documented in [`DATA.md`](DATA.md).

Before committing new changes, verify that no private data or environment-specific secrets have been added.

---

## Security Checklist

Before pushing changes:

- confirm `data/*.jsonl` is not committed;
- confirm `.env` is not committed;
- remove API keys and access tokens;
- remove internal hostnames and service URLs;
- remove private model or dataset paths;
- review generated logs and outputs before committing.

The repository includes `.gitignore` and `.env.example` to support this workflow.

---

## Scope

This repository is intended as a **reusable fine-tuning and serving template**, not as a public release of the original internship dataset or business system.

It demonstrates the engineering workflow for:

- instruction-dataset preparation;
- parameter-efficient LLM adaptation;
- QLoRA / LoRA training;
- DeepSpeed-based training configuration;
- adapter-based model deployment;
- vLLM serving.

Domain-specific data processing logic, private evaluation sets, proprietary knowledge bases, and internal production infrastructure are outside the scope of this public repository.

---

## Technology Stack

| Area | Tools |
| --- | --- |
| Model Training | PyTorch, Hugging Face Transformers |
| Parameter-Efficient Tuning | PEFT, LoRA, QLoRA |
| Training Framework | LLaMA-Factory |
| Distributed Training | DeepSpeed ZeRO-2 |
| Inference Serving | vLLM |
| Data Format | Alpaca-style JSONL |
| Environment | Python, Bash |

---

## Internship Context

This repository was prepared from reusable components developed during an AI application internship.

The public version is intentionally decoupled from the original domain data and internal infrastructure. Its purpose is to preserve the **generalizable LLM engineering workflow** while respecting privacy and compliance constraints.

The internship work covered a broader domain-adaptation pipeline; this repository specifically exposes the reusable **SFT / QLoRA training and vLLM serving components** that can be safely published.

---

## License

This repository is released under the MIT License.

See [`LICENSE`](LICENSE) for details.
