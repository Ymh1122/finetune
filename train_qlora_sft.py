import argparse
import json
import math
import os
import inspect
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.utils.checkpoint as torch_checkpoint
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def build_prompt(instruction: str, user_input: str) -> str:
    instruction = (instruction or "").strip()
    user_input = (user_input or "").strip()
    if user_input:
        return f"### 指令\n{instruction}\n\n### 输入\n{user_input}\n\n### 回复\n"
    return f"### 指令\n{instruction}\n\n### 回复\n"


def tokenize_and_mask(
    tokenizer: Any,
    instruction: str,
    user_input: str,
    output: str,
    max_length: int,
) -> Dict[str, Any]:
    prompt = build_prompt(instruction, user_input)
    answer = (output or "").strip()

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer, add_special_tokens=False).input_ids

    # Append EOS
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        answer_ids = answer_ids + [eos_id]

    input_ids = prompt_ids + answer_ids
    input_ids = input_ids[:max_length]

    labels = [-100] * min(len(prompt_ids), max_length) + input_ids[min(len(prompt_ids), max_length) :]
    labels = labels[:max_length]

    attention_mask = [1] * len(input_ids)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@dataclass
class DataCollatorForCausalLM:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            ids = f["input_ids"]
            mask = f["attention_mask"]
            lab = f["labels"]

            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append(mask + [0] * pad_len)
            labels.append(lab + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main() -> None:
    if sys.version_info >= (3, 12):
        print(
            "[warn] Detected Python {}.{}. Training stacks (PyTorch/bitsandbytes) often do not provide wheels for such new versions. "
            "If you hit install/import errors, use Python 3.10/3.11.".format(sys.version_info.major, sys.version_info.minor)
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--eval_file", default=None)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target",
        type=str,
        # MoE models often contain per-expert MLP projections (e.g. *.experts.*.up_proj/down_proj).
        # LoRA-ing those can trigger DDP corner cases (unused params or re-entrant checkpointing issues).
        # Default to attention projections only; expand explicitly if you know what you're doing.
        default="q_proj,k_proj,v_proj,o_proj",
    )

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    # For small SFT datasets, too-large global batch can collapse steps/epoch.
    # Default to a smaller accumulation so the run actually has enough optimizer steps.
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    # QLoRA on a large base model is usually stable in the 5e-5~2e-4 range.
    # With small data, prefer a slightly lower LR and train for more epochs.
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=float, default=4.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument(
        "--optim",
        type=str,
        default="paged_adamw_8bit",
        help="If supported by your transformers version, use bitsandbytes paged AdamW for QLoRA.",
    )

    parser.add_argument("--logging_steps", type=int, default=5)
    # Small datasets: save more frequently so you can pick the best checkpoint.
    parser.add_argument("--save_steps", type=int, default=20)
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=20,
        help="When eval_file is provided, run evaluation every N steps.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="If >0, override num_train_epochs and train for a fixed number of optimizer steps.",
    )
    parser.add_argument(
        "--load_best_model_at_end",
        default=True,
        action=getattr(argparse, "BooleanOptionalAction", "store_true"),
        help="When eval_file is provided, keep the best checkpoint by eval_loss.",
    )
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)

    # For MoE / sparse-routing models, many parameters won't receive grads every step.
    # DDP must be told to tolerate this, otherwise it can throw the "finished reduction" error.
    parser.add_argument(
        "--ddp_find_unused_parameters",
        # With attention-only LoRA, all trainable params should be used every step.
        # Keep this False by default to avoid DDP+checkpointing edge cases.
        default=False,
        action=getattr(argparse, "BooleanOptionalAction", "store_true"),
    )

    parser.add_argument(
        "--gradient_checkpointing",
        default=True,
        action=getattr(argparse, "BooleanOptionalAction", "store_true"),
    )
    parser.add_argument("--bf16", action="store_true", default=True)

    args = parser.parse_args()

    # torchrun sets LOCAL_RANK/RANK/WORLD_SIZE. Bind each process to one GPU.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        # For many causal LMs, set pad to eos
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        quantization_config=bnb_config,
        # IMPORTANT: with torchrun/DDP, each rank must stick to its own GPU.
        # device_map="auto" may place multiple ranks on GPU0 and cause OOM.
        device_map={"": local_rank} if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )

    if args.gradient_checkpointing:
        # Some model implementations call torch.utils.checkpoint.checkpoint without specifying
        # use_reentrant; PyTorch warns now and DDP can error with "marked ready twice".
        # Force non-reentrant checkpointing unless the model explicitly sets it.
        try:
            _orig_checkpoint = torch_checkpoint.checkpoint

            def _checkpoint_no_reentrant(function, *c_args, **c_kwargs):
                if "use_reentrant" not in c_kwargs:
                    c_kwargs["use_reentrant"] = False
                return _orig_checkpoint(function, *c_args, **c_kwargs)

            torch_checkpoint.checkpoint = _checkpoint_no_reentrant  # type: ignore[assignment]
        except Exception:
            pass

        # Disable KV cache when using gradient checkpointing.
        if getattr(model, "config", None) is not None:
            model.config.use_cache = False

        # PyTorch 2.9 will require explicitly setting use_reentrant.
        # Prefer use_reentrant=False for better compatibility.
        try:
            sig = inspect.signature(model.gradient_checkpointing_enable)
            if "gradient_checkpointing_kwargs" in sig.parameters:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            else:
                model.gradient_checkpointing_enable()
        except Exception:
            model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(model)

    lora_target_modules = [s.strip() for s in args.lora_target.split(",") if s.strip()]
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_target_modules,
    )
    model = get_peft_model(model, lora_config)

    data_files = {"train": args.train_file}
    if args.eval_file:
        data_files["validation"] = args.eval_file

    ds = load_dataset("json", data_files=data_files)

    def map_fn(ex: Dict[str, Any]) -> Dict[str, Any]:
        return tokenize_and_mask(
            tokenizer=tokenizer,
            instruction=ex.get("instruction", ""),
            user_input=ex.get("input", ""),
            output=ex.get("output", ""),
            max_length=args.max_length,
        )

    train_ds = ds["train"].map(map_fn, remove_columns=ds["train"].column_names)
    eval_ds = None
    if "validation" in ds:
        eval_ds = ds["validation"].map(map_fn, remove_columns=ds["validation"].column_names)

    # evaluation/save strategy should match when load_best_model_at_end=True.
    do_eval = eval_ds is not None

    training_args = TrainingArguments(
        **(lambda: {
            k: v
            for k, v in {
                "output_dir": args.output_dir,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "num_train_epochs": args.num_train_epochs,
                "max_steps": args.max_steps if args.max_steps and args.max_steps > 0 else None,
                "warmup_ratio": args.warmup_ratio,
                "weight_decay": args.weight_decay,
                "max_grad_norm": args.max_grad_norm,
                "lr_scheduler_type": args.lr_scheduler_type,
                "optim": args.optim,
                "logging_steps": args.logging_steps,
                "save_steps": args.save_steps,
                "save_total_limit": args.save_total_limit,
                "bf16": args.bf16,
                "fp16": False,
                "report_to": [],
                "remove_unused_columns": False,
                # Newer transformers: evaluation_strategy; some versions: eval_strategy; older: none.
                "evaluation_strategy": "no" if not do_eval else "steps",
                "eval_strategy": "no" if not do_eval else "steps",
                "eval_steps": args.eval_steps if do_eval else None,
                "load_best_model_at_end": (args.load_best_model_at_end if do_eval else False),
                "metric_for_best_model": ("eval_loss" if do_eval else None),
                "greater_is_better": (False if do_eval else None),
                "ddp_find_unused_parameters": args.ddp_find_unused_parameters,
                "seed": args.seed,
            }.items()
            if k in set(inspect.signature(TrainingArguments.__init__).parameters)
            and v is not None
        })()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForCausalLM(pad_token_id=tokenizer.pad_token_id),
    )

    trainer.train()

    # Save adapter + tokenizer
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
