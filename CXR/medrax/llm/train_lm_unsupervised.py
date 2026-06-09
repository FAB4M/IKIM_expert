"""Unsupervised LoRA continued-pretraining von Qwen auf Befund-Text (next-token, keine Labels).
fp16-LoRA (Default), optional --4bit (QLoRA).

    python -m medrax.llm.train_lm_unsupervised --epochs 1 --base-model Qwen/Qwen2.5-1.5B-Instruct
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config  # noqa: E402  (setzt KMP_DUPLICATE_LIB_OK etc.)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Unsupervised LoRA continued-pretraining of Qwen on reports")
    p.add_argument("--corpus", default=str(config.REPORTS_CORPUS))
    p.add_argument("--base-model", default=config.BASE_LLM)
    p.add_argument("--out", default=str(config.LLM_ADAPTER_DIR))
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--max-samples", type=int, default=None, help="Korpus auf N Beispiele begrenzen")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--4bit", dest="four_bit", action="store_true", help="QLoRA (bitsandbytes)")
    return p.parse_args(argv)


def _require(mod, pipname=None):
    try:
        return __import__(mod)
    except Exception as e:
        raise SystemExit(
            f"[FEHLER] Paket '{mod}' fehlt. Bitte installieren:\n"
            f"    pip install {pipname or mod}\n(Detail: {e})"
        )


def main(argv=None):
    args = parse_args(argv)
    _require("torch")
    _require("transformers")
    _require("peft")
    _require("datasets")

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForLanguageModeling, Trainer, TrainingArguments)

    if not os.path.exists(args.corpus):
        raise SystemExit(
            f"[FEHLER] Korpus nicht gefunden: {args.corpus}\n"
            "-> Zuerst:  python -m medrax.llm.prepare_reports --max-samples 20000"
        )

    print(f"[train-lm] Basis-Modell: {args.base_model}")
    print(f"[train-lm] Korpus: {args.corpus}")
    print(f"[train-lm] Modus: {'QLoRA 4-bit' if args.four_bit else 'fp16 LoRA'}")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Modell ---
    if args.four_bit:
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, quantization_config=bnb, device_map="auto", trust_remote_code=True)
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    except Exception:
        pass

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # --- Daten ---
    ds = load_dataset("json", data_files=args.corpus, split="train")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print(f"[train-lm] Beispiele: {len(ds)}")

    eos = tokenizer.eos_token or ""

    def tok(batch):
        texts = [t + eos for t in batch["text"]]
        return tokenizer(texts, truncation=True, max_length=args.max_seq_len)

    ds = ds.map(tok, batched=True, remove_columns=ds.column_names)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    targs = TrainingArguments(
        output_dir=os.path.join(args.out, "trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=25,
        save_strategy="no",
        fp16=torch.cuda.is_available() and not args.four_bit,
        gradient_checkpointing=True,
        report_to="none",
        optim="paged_adamw_8bit" if args.four_bit else "adamw_torch",
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    print("[train-lm] Training startet ...")
    trainer.train()

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)      # nur LoRA-Adapter
    tokenizer.save_pretrained(args.out)
    print(f"\n[train-lm] LoRA-Adapter gespeichert: {args.out}")
    print("-> Wird beim Servieren (serve_qwen.py --adapter ...) auf das Basismodell geladen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
