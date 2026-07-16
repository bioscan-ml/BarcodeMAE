import csv
import json
import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import transformers
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from transformers import DataCollatorWithPadding, Trainer, set_seed

from bert_layers import BertForSequenceClassification
from maelm_model import MAELMForSequenceClassification


LOGGER = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Model checkpoint or Hugging Face model name."})
    model_type: str = field(default="auto", metadata={"help": "Model type: auto, bert, dnabert, or maelm."})
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Optional tokenizer name or path. Defaults to the model name for InstaDeep models."},
    )
    use_lora: bool = field(default=True, metadata={"help": "Apply LoRA adapters, matching the notebook recipe."})
    lora_r: int = field(default=1, metadata={"help": "LoRA rank from the notebook recipe."})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha from the notebook recipe."})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout from the notebook recipe."})
    lora_target_modules: str = field(
        default="query,value",
        metadata={"help": "Comma-separated target module names for LoRA."},
    )


@dataclass
class BenchmarkArguments:
    run_name: str = field(metadata={"help": "Benchmark run name used for output directories and CSV files."})
    dataset_name: str = field(
        default="InstaDeepAI/nucleotide_transformer_downstream_tasks_revised",
        metadata={"help": "Hugging Face dataset containing the revised Nucleotide Transformer benchmark."},
    )
    task_names: Optional[str] = field(
        default=None,
        metadata={"help": "Optional comma-separated task list. Defaults to every task in the train split."},
    )
    output_dir: str = field(
        default="output/nt_benchmark",
        metadata={"help": "Root directory where fold outputs and aggregate CSV results are written."},
    )
    n_splits: int = field(default=10, metadata={"help": "Number of stratified cross-validation folds."})


@dataclass
class TrainingConfig:
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(default=1024, metadata={"help": "Maximum sequence length passed to the tokenizer."})
    learning_rate: float = field(default=5e-4)
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=64)
    gradient_accumulation_steps: int = field(default=1)
    num_train_epochs: float = field(default=2.0)
    max_steps: int = field(default=10000)
    logging_steps: int = field(default=1000)
    eval_steps: int = field(default=1000)
    save_steps: int = field(default=1000)
    warmup_steps: int = field(default=0)
    weight_decay: float = field(default=0.0)
    dataloader_num_workers: int = field(default=0)
    fp16: bool = field(default=False)
    bf16: bool = field(default=False)
    seed: int = field(default=42)
    use_wandb: bool = field(default=False)
    wandb_project: str = field(default="nt-benchmark")
    wandb_entity: Optional[str] = field(default=None)
    wandb_group: Optional[str] = field(default=None)
    wandb_tags: str = field(default="finetuning,nt-benchmark")
    overwrite_output_dir: bool = field(default=True)


def parse_task_names(task_names: Optional[str]) -> Optional[List[str]]:
    if not task_names:
        return None
    parsed = [task_name.strip() for task_name in task_names.split(",") if task_name.strip()]
    return parsed or None


def parse_wandb_tags(tags: str) -> List[str]:
    return [tag.strip() for tag in tags.split(",") if tag.strip()]


def resolve_column(columns: Sequence[str], candidates: Sequence[str], label: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not resolve the {label} column. Available columns: {list(columns)}")


def compute_metrics_mcc(eval_pred: transformers.EvalPrediction) -> Dict[str, float]:
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    references = eval_pred.label_ids
    return {"mcc_score": matthews_corrcoef(references, predictions)}


def build_tokenizer(model_args: ModelArguments, training_config: TrainingConfig) -> transformers.PreTrainedTokenizer:
    if model_args.tokenizer_name:
        tokenizer_name = model_args.tokenizer_name
    elif "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer_name = model_args.model_name_or_path
    else:
        tokenizer_name = "zhihan1996/DNABERT-2-117M"

    LOGGER.info("Loading tokenizer from %s", tokenizer_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=training_config.cache_dir,
        model_max_length=training_config.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
    return tokenizer


def load_maelm_model(model_args: ModelArguments, training_config: TrainingConfig, num_labels: int):
    if os.path.exists(model_args.model_name_or_path):
        config_path = model_args.model_name_or_path
    else:
        config_path = "zhihan1996/DNABERT-2-117M"

    config = transformers.AutoConfig.from_pretrained(
        config_path,
        num_labels=num_labels,
        cache_dir=training_config.cache_dir,
        trust_remote_code=True,
    )

    try:
        model = MAELMForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_config.cache_dir,
            ignore_mismatched_sizes=True,
        )
    except Exception as exc:
        LOGGER.warning("Automatic MAELM loading failed: %s", exc)
        model = MAELMForSequenceClassification(config)

        if os.path.isdir(model_args.model_name_or_path):
            checkpoint_path = os.path.join(model_args.model_name_or_path, "pytorch_model.bin")
        else:
            checkpoint_path = model_args.model_name_or_path

        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]

            remapped_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("bert."):
                    remapped_state_dict[key.replace("bert.", "encoder.")] = value
                else:
                    remapped_state_dict[key] = value

            missing, unexpected = model.load_state_dict(remapped_state_dict, strict=False)
            LOGGER.info("Manual MAELM load complete. Missing=%d Unexpected=%d", len(missing), len(unexpected))

    return model


def load_bert_model(model_args: ModelArguments, training_config: TrainingConfig, num_labels: int):
    if os.path.exists(model_args.model_name_or_path):
        if os.path.isdir(model_args.model_name_or_path):
            try:
                return BertForSequenceClassification.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_config.cache_dir,
                    num_labels=num_labels,
                    trust_remote_code=True,
                )
            except Exception as exc:
                LOGGER.warning("Falling back to config-based BERT loading: %s", exc)
                try:
                    config = transformers.BertConfig.from_pretrained(model_args.model_name_or_path)
                except Exception:
                    config = transformers.BertConfig(
                        vocab_size=4096,
                        hidden_size=768,
                        num_hidden_layers=12,
                        num_attention_heads=12,
                        max_position_embeddings=512,
                    )
                config.num_labels = num_labels
                if not hasattr(config, "alibi_starting_size"):
                    config.alibi_starting_size = 512
                return BertForSequenceClassification(config)

        checkpoint = torch.load(model_args.model_name_or_path, map_location="cpu")
        if "config" in checkpoint:
            config = transformers.BertConfig.from_dict(checkpoint["config"])
        else:
            config = transformers.BertConfig(
                vocab_size=4096,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                max_position_embeddings=512,
                alibi_starting_size=512,
            )
        config.num_labels = num_labels
        model = BertForSequenceClassification(config)

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        remapped_state_dict = {}
        for key, value in state_dict.items():
            remapped_state_dict[key[7:] if key.startswith("module.") else key] = value

        missing = model.load_state_dict(remapped_state_dict, strict=False)
        LOGGER.info(
            "Manual BERT load complete. Missing=%d Unexpected=%d",
            len(missing.missing_keys),
            len(missing.unexpected_keys),
        )
        return model

    return transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_config.cache_dir,
        num_labels=num_labels,
        trust_remote_code=True,
    )


def build_model(
    model_args: ModelArguments,
    training_config: TrainingConfig,
    num_labels: int,
    pad_token_id: Optional[int],
):
    model_type = model_args.model_type.lower()
    if model_type == "maelm":
        model = load_maelm_model(model_args, training_config, num_labels)
    elif model_type in {"bert", "dnabert"}:
        model = load_bert_model(model_args, training_config, num_labels)
    else:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_config.cache_dir,
            num_labels=num_labels,
            trust_remote_code=True,
        )

    if getattr(model.config, "pad_token_id", None) is None and pad_token_id is not None:
        model.config.pad_token_id = pad_token_id

    if model_args.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=[module.strip() for module in model_args.lora_target_modules.split(",") if module.strip()],
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model


def build_task_dataset(task_dataset: Dataset, sequence_column: str, label_column: str) -> Dataset:
    return Dataset.from_dict(
        {
            "sequence": [str(sequence) for sequence in task_dataset[sequence_column]],
            "labels": [int(label) for label in task_dataset[label_column]],
        }
    )


def tokenize_task_dataset(
    task_dataset: Dataset,
    tokenizer: transformers.PreTrainedTokenizer,
    model_max_length: int,
) -> Dataset:
    def tokenize_batch(examples: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        return tokenizer(examples["sequence"], truncation=True, max_length=model_max_length)

    tokenized = task_dataset.map(tokenize_batch, batched=True, remove_columns=["sequence"])
    torch_columns = [column for column in ["input_ids", "attention_mask", "token_type_ids", "labels"] if column in tokenized.column_names]
    tokenized.set_format(type="torch", columns=torch_columns)
    return tokenized


def build_training_arguments(
    benchmark_args: BenchmarkArguments,
    training_config: TrainingConfig,
    task_name: str,
    fold_index: int,
) -> transformers.TrainingArguments:
    output_dir = os.path.join(benchmark_args.output_dir, benchmark_args.run_name, task_name, f"fold_{fold_index + 1}")
    report_to = ["wandb"] if training_config.use_wandb else []

    return transformers.TrainingArguments(
        output_dir=output_dir,
        run_name=f"{benchmark_args.run_name}_{task_name}_fold{fold_index + 1}",
        remove_unused_columns=False,
        eval_strategy="steps",
        save_strategy="steps",
        learning_rate=training_config.learning_rate,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        per_device_eval_batch_size=training_config.per_device_eval_batch_size,
        num_train_epochs=training_config.num_train_epochs,
        logging_steps=training_config.logging_steps,
        eval_steps=training_config.eval_steps,
        save_steps=training_config.save_steps,
        warmup_steps=training_config.warmup_steps,
        weight_decay=training_config.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="mcc_score",
        greater_is_better=True,
        label_names=["labels"],
        dataloader_drop_last=True,
        dataloader_num_workers=training_config.dataloader_num_workers,
        max_steps=training_config.max_steps,
        fp16=training_config.fp16,
        bf16=training_config.bf16,
        report_to=report_to,
        save_total_limit=1,
        overwrite_output_dir=training_config.overwrite_output_dir,
        seed=training_config.seed + fold_index,
        data_seed=training_config.seed + fold_index,
    )


def run_single_fold(
    model_args: ModelArguments,
    benchmark_args: BenchmarkArguments,
    training_config: TrainingConfig,
    tokenizer: transformers.PreTrainedTokenizer,
    tokenized_task_dataset: Dataset,
    task_name: str,
    fold_index: int,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
    num_labels: int,
) -> Dict[str, float]:
    set_seed(training_config.seed + fold_index)
    model = build_model(model_args, training_config, num_labels, tokenizer.pad_token_id)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataset = tokenized_task_dataset.select(train_indices.tolist())
    eval_dataset = tokenized_task_dataset.select(eval_indices.tolist())

    fold_training_args = build_training_arguments(benchmark_args, training_config, task_name, fold_index)
    if training_config.use_wandb:
        os.environ.setdefault("WANDB_PROJECT", training_config.wandb_project)
        if training_config.wandb_entity:
            os.environ.setdefault("WANDB_ENTITY", training_config.wandb_entity)
        if training_config.wandb_group:
            os.environ["WANDB_RUN_GROUP"] = training_config.wandb_group
        if training_config.wandb_tags:
            os.environ["WANDB_TAGS"] = ",".join(parse_wandb_tags(training_config.wandb_tags))

    trainer = Trainer(
        model=model,
        args=fold_training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_mcc,
    )
    trainer.train()
    eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)

    os.makedirs(fold_training_args.output_dir, exist_ok=True)
    with open(os.path.join(fold_training_args.output_dir, "fold_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(eval_metrics, handle, indent=2)

    eval_metrics["mcc_x100"] = 100.0 * eval_metrics["eval_mcc_score"]
    return eval_metrics


def validate_class_balance(labels: Sequence[int], n_splits: int, task_name: str) -> None:
    label_counts = Counter(labels)
    min_count = min(label_counts.values())
    if min_count < n_splits:
        raise ValueError(
            f"Task '{task_name}' cannot support {n_splits}-fold stratified CV because the smallest class only has {min_count} samples."
        )


def write_results_csv(csv_path: str, rows: List[Dict[str, object]], n_splits: int) -> None:
    fieldnames = ["task", "num_samples", "num_labels"]
    fieldnames.extend([f"fold_{fold_index + 1}_mcc_x100" for fold_index in range(n_splits)])
    fieldnames.extend(
        [
            "mcc_x100_mean",
            "mcc_x100_median",
            "mcc_x100_std",
            "mcc_x100_summary",
            "mcc_x100_summary_median",
        ]
    )

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = transformers.HfArgumentParser((ModelArguments, BenchmarkArguments, TrainingConfig))
    model_args, benchmark_args, training_config = parser.parse_args_into_dataclasses()

    requested_tasks = parse_task_names(benchmark_args.task_names)
    tokenizer = build_tokenizer(model_args, training_config)
    dataset = load_dataset(benchmark_args.dataset_name, cache_dir=training_config.cache_dir)
    if "train" not in dataset:
        raise ValueError(f"Dataset '{benchmark_args.dataset_name}' does not expose a train split.")

    train_split = dataset["train"]
    sequence_column = resolve_column(train_split.column_names, ["sequence", "data", "seq"], "sequence")
    label_column = resolve_column(train_split.column_names, ["label", "labels", "target"], "label")
    task_column = resolve_column(train_split.column_names, ["task", "dataset_name", "subset"], "task")

    available_tasks = sorted(set(train_split[task_column]))
    task_names = requested_tasks or available_tasks
    missing_tasks = [task_name for task_name in task_names if task_name not in available_tasks]
    if missing_tasks:
        raise ValueError(f"Requested tasks not found in the train split: {missing_tasks}. Available tasks: {available_tasks}")

    results_rows: List[Dict[str, object]] = []
    for task_name in task_names:
        LOGGER.info("Running %d-fold CV for task %s", benchmark_args.n_splits, task_name)
        task_train_split = train_split.filter(lambda example: example[task_column] == task_name)
        task_dataset = build_task_dataset(task_train_split, sequence_column, label_column)
        labels = np.asarray(task_dataset["labels"])
        validate_class_balance(labels, benchmark_args.n_splits, task_name)

        tokenized_task_dataset = tokenize_task_dataset(task_dataset, tokenizer, training_config.model_max_length)
        splitter = StratifiedKFold(
            n_splits=benchmark_args.n_splits,
            shuffle=True,
            random_state=training_config.seed,
        )
        num_labels = len(set(task_dataset["labels"]))

        fold_scores: List[float] = []
        row: Dict[str, object] = {
            "task": task_name,
            "num_samples": len(task_dataset),
            "num_labels": num_labels,
        }

        dummy_features = np.zeros(len(labels))
        for fold_index, (train_indices, eval_indices) in enumerate(splitter.split(dummy_features, labels)):
            fold_metrics = run_single_fold(
                model_args=model_args,
                benchmark_args=benchmark_args,
                training_config=training_config,
                tokenizer=tokenizer,
                tokenized_task_dataset=tokenized_task_dataset,
                task_name=task_name,
                fold_index=fold_index,
                train_indices=train_indices,
                eval_indices=eval_indices,
                num_labels=num_labels,
            )
            fold_score = float(fold_metrics["mcc_x100"])
            fold_scores.append(fold_score)
            row[f"fold_{fold_index + 1}_mcc_x100"] = f"{fold_score:.4f}"
            LOGGER.info("Task %s fold %d MCC x100 = %.4f", task_name, fold_index + 1, fold_score)

        mean_score = float(np.mean(fold_scores))
        median_score = float(np.median(fold_scores))
        std_score = float(np.std(fold_scores))
        row["mcc_x100_mean"] = f"{mean_score:.4f}"
        row["mcc_x100_median"] = f"{median_score:.4f}"
        row["mcc_x100_std"] = f"{std_score:.4f}"
        row["mcc_x100_summary"] = f"{mean_score:.2f} ± {std_score:.2f}"
        row["mcc_x100_summary_median"] = f"{median_score:.2f} ± {std_score:.2f}"
        results_rows.append(row)

    results_dir = os.path.join(benchmark_args.output_dir, benchmark_args.run_name)
    csv_path = os.path.join(results_dir, "nt_benchmark_cv_results.csv")
    write_results_csv(csv_path, results_rows, benchmark_args.n_splits)

    summary_path = os.path.join(results_dir, "nt_benchmark_cv_results.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(results_rows, handle, indent=2)

    LOGGER.info("Saved benchmark summary CSV to %s", csv_path)


if __name__ == "__main__":
    main()
