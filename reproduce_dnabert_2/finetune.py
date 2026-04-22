import os
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union

import torch
import transformers
import sklearn
import numpy as np
from torch.utils.data import Dataset
from bert_layers import BertForSequenceClassification

from maelm_model import MAELMModel, MAELMForSequenceClassification

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        get_peft_model_state_dict,
        TaskType,
    )
except ImportError:
    pass


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_type: str = field(default="bert", metadata={"help": "Model type: 'bert' or 'maelm'"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps")
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=True)
    seed: int = field(default=42)
    evaluation_strategy: str = "no"
    save_strategy: str = 'no' 

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


"""
Get the reversed complement of the original DNA sequence.
"""
def get_alter_of_dna_sequence(sequence: str):
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    # return "".join([MAP[c] for c in reversed(sequence)])
    return "".join([MAP[c] for c in sequence])

"""
Transform a dna sequence to k-mer string
"""
def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])


"""
Load or generate k-mer string for each DNA sequence. The generated k-mer string will be saved to the same directory as the original data with the same name but with a suffix of "_{k}mer".
"""
def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each DNA sequence."""
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:    
        logging.warning(f"Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            json.dump(kmer, f)
    return kmer

class SupervisedDataset(Dataset):
    """Dataset for supervised learning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, kmer: int = -1):
        super(SupervisedDataset, self).__init__()

        # load data from the disk
        print(f"Loading data from {data_path}")
        with open(data_path, "r") as f:
            # Check for delimiter by trying tab first
            reader = csv.reader(f, delimiter="\t")
            try:
                data = list(reader)[1:]
            except Exception as e:
                print(f"Error reading with tab delimiter: {e}")
                data = []

        if not data or len(data[0]) <= 1:
            # Try comma delimiter
            print("Tab delimiter failed or single column, trying comma...")
            with open(data_path, "r") as f:
                reader = csv.reader(f, delimiter=",")
                data = list(reader)[1:]

        if len(data) == 0:
            raise ValueError(f"Data file {data_path} is empty or unreadable.")

        if len(data[0]) == 2:
            # data is in the format of [text, label]
            logging.warning("Perform single sequence classification...")
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            logging.warning("Perform sequence-pair classification...")
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        else:
            print(f"First row: {data[0]}")
            raise ValueError("Data format not supported. Expected 2 or 3 columns (text, label) or (text1, text2, label).")
        
        if kmer != -1:
            # only write file on the first process
            if torch.distributed.is_initialized():
                if torch.distributed.get_rank() == 0:
                    load_or_generate_kmer(data_path, texts, kmer)
                torch.distributed.barrier()
            else:
                 load_or_generate_kmer(data_path, texts, kmer)

            texts = load_or_generate_kmer(data_path, texts, kmer)

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.num_labels = len(set(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i], attention_mask=self.attention_mask[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised learning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        attention_mask = [instance["attention_mask"] for instance in instances]
        labels = [instance["labels"] for instance in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.tensor(labels, dtype=torch.long)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""
def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray):
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
    }

# from: https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13
def preprocess_logits_for_metrics(logits:Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]

    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])

    return torch.argmax(logits, dim=-1)


"""
Compute metrics used for huggingface trainer.
""" 
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return calculate_metric_with_sklearn(predictions, labels)



def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # load tokenizer
    # Always default to valid DNABERT-2 tokenizer unless explicitly overridden with a valid tokenizer name
    if model_args.tokenizer_name:
        tokenizer_name = model_args.tokenizer_name
    else:
        # Fallback to base tokenizer
        tokenizer_name = "zhihan1996/DNABERT-2-117M"

    print(f"Loading tokenizer from: {tokenizer_name}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    if "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token

    # define datasets and data collator
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=os.path.join(data_args.data_path, "train.csv"), kmer=data_args.kmer)
    val_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=os.path.join(data_args.data_path, "dev.csv"), kmer=data_args.kmer)
    test_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=os.path.join(data_args.data_path, "test.csv"), kmer=data_args.kmer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # load model
    num_labels = train_dataset.num_labels
    print(f"Number of labels: {num_labels}")

    if model_args.model_type == "maelm":
        print("Loading MAELM checkpoint for classification...")
        # MAELM checkpoints have 'encoder.' prefix for the BERT part
        # We need to load it into BertForSequenceClassification
        # 1. Load config
        if os.path.exists(model_args.model_name_or_path):
            config_path = model_args.model_name_or_path
        else:
            config_path = "zhihan1996/DNABERT-2-117M" # Fallback if just loading from weights file without config.json

        config = transformers.AutoConfig.from_pretrained(
            config_path,
            num_labels=num_labels,
            finetuning_task=data_args.data_path,
            cache_dir=training_args.cache_dir,
            trust_remote_code=True,
        )
        
        # 2. Initialize MAELM classification model
        # Use custom MAELMForSequenceClassification
        # This wrapper ensures encoder keys are loaded correctly and adds a classification head
        try:
            model = MAELMForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                ignore_mismatched_sizes=True, 
            )
        except Exception as e:
            print(f"Warning: automatic loading failed: {e}")
            print("Attempting manual load...")
            model = MAELMForSequenceClassification(config)
            
            if os.path.isdir(model_args.model_name_or_path):
                ckpt_file = os.path.join(model_args.model_name_or_path, "pytorch_model.bin")
            else:
                ckpt_file = model_args.model_name_or_path
                
            if os.path.exists(ckpt_file):
                print(f"Loading weights manually from {ckpt_file}")
                state_dict = torch.load(ckpt_file, map_location="cpu")
                if "model" in state_dict: state_dict = state_dict["model"]
                
                # Check for 'encoder.' keys (MAELM format) vs 'bert.' keys (DNABERT format)
                # MAELMForSequenceClassification uses 'encoder', so if checkpoint has 'bert', rename.
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("bert."):
                        new_state_dict[k.replace("bert.", "encoder.")] = v
                    else:
                        new_state_dict[k] = v
                        
                missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
                print(f"Manual Load | Missing: {len(missing)} | Unexpected: {len(unexpected)}")
            else:
                print(f"Checkpoint file not found at {ckpt_file}, using random initialization based on config.")
            
    elif model_args.model_type == "dnabert" or model_args.model_type == "bert":
        if os.path.exists(model_args.model_name_or_path):
            print("The model exists! ")
            if os.path.isdir(model_args.model_name_or_path):
                try:
                    model = BertForSequenceClassification.from_pretrained(
                        model_args.model_name_or_path,
                        cache_dir=training_args.cache_dir,
                        num_labels=num_labels,
                        trust_remote_code=True,
                    )
                except Exception as e:
                    logging.warning(f"Could not load as HF model, creating from scratch using config: {e}")
                    # Try to load config from the directory
                    try:
                        config = transformers.BertConfig.from_pretrained(model_args.model_name_or_path)
                    except:
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
                    model = BertForSequenceClassification(config)
            else:
                # Assuming it is a checkpoint from train.py
                print(f"Loading from local checkpoint: {model_args.model_name_or_path}")
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
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("module."):
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v

                keys = model.load_state_dict(new_state_dict, strict=False)
                logging.info(f"Model loaded. Missing keys: {keys.missing_keys}. Unexpected keys: {keys.unexpected_keys}")

        else:
            print("Model path not found, downloading from HF")
            # load model
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                num_labels=num_labels,
                trust_remote_code=True,
            )
    else:
        raise ValueError(f"Unknown model type: {model_args.model_type}")
    
    # Configure LoRA if requested
    if model_args.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules.split(","),
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # define trainer
    trainer = transformers.Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        data_collator=data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()

    # save model
    if training_args.save_model:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # evaluate
    if training_args.eval_and_save_results and val_dataset is not None:
        results = trainer.evaluate(eval_dataset=val_dataset)
        with open(os.path.join(training_args.output_dir,training_args.run_name,"eval_results.json"), "w") as f:
            json.dump(results, f)
            
    if test_dataset is not None:
         predictions = trainer.predict(test_dataset)
         with open(os.path.join(training_args.output_dir,training_args.run_name,"test_results.json"), "w") as f:
             json.dump(predictions.metrics, f)

if __name__ == "__main__":
    train()
