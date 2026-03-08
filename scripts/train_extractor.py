#!/usr/bin/env python3
"""Train a self-supervised fact extraction model from collected LLM extractions.

Usage:
    python scripts/train_extractor.py --data extractions.jsonl --output models/extractor

Requires: pip install transformers datasets torch
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_data(path: str):
    samples = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            input_text = item["input"]
            output = json.dumps({"facts": item["output"]})
            samples.append({"input": input_text, "output": output})
    return samples


def train(data_path: str, output_dir: str, base_model: str, epochs: int, batch_size: int):
    try:
        from datasets import Dataset
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
    except ImportError:
        print("Error: Install training dependencies:")
        print("  pip install transformers datasets torch")
        return

    samples = load_data(data_path)
    print(f"Loaded {len(samples)} training samples")

    if len(samples) < 50:
        print(f"Warning: Only {len(samples)} samples. Recommend at least 500 for good quality.")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

    def preprocess(examples):
        inputs = tokenizer(
            examples["input"],
            max_length=512,
            truncation=True,
            padding="max_length",
        )
        targets = tokenizer(
            examples["output"],
            max_length=256,
            truncation=True,
            padding="max_length",
        )
        inputs["labels"] = targets["input_ids"]
        return inputs

    dataset = Dataset.from_list(samples)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"].map(preprocess, batched=True, remove_columns=["input", "output"])
    eval_ds = split["test"].map(preprocess, batched=True, remove_columns=["input", "output"])

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        predict_with_generate=True,
        fp16=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


def export_data(db_path: str, output_path: str):
    from widemem.extraction.collector import ExtractionCollector
    collector = ExtractionCollector(db_path)
    count = collector.export(output_path)
    collector.close()
    print(f"Exported {count} samples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train widemem fact extraction model")
    sub = parser.add_subparsers(dest="command")

    export_parser = sub.add_parser("export", help="Export training data from collector DB")
    export_parser.add_argument("--db", default="~/.widemem/extractions.db")
    export_parser.add_argument("--output", default="extractions.jsonl")

    train_parser = sub.add_parser("train", help="Train the extraction model")
    train_parser.add_argument("--data", required=True, help="Path to JSONL training data")
    train_parser.add_argument("--output", default="models/extractor", help="Output model directory")
    train_parser.add_argument("--base-model", default="google/flan-t5-small", help="Base model to fine-tune")
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--batch-size", type=int, default=8)

    args = parser.parse_args()

    if args.command == "export":
        export_data(args.db, args.output)
    elif args.command == "train":
        train(args.data, args.output, args.base_model, args.epochs, args.batch_size)
    else:
        parser.print_help()
