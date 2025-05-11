import os
import json
import argparse
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import torch

# ----------------------------
# Argument Parser
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--quantized", action="store_true")
parser.add_argument("--deepspeed", type=str, default=None)
parser.add_argument("--hub_repo", type=str, default=None)
parser.add_argument("--local_rank", type=int, default=-1, help="Used by deepspeed")
args = parser.parse_args()

# ----------------------------
# Load tokenizer and model
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
tokenizer.use_default_system_prompt = False
tokenizer.pad_token = tokenizer.eos_token 

model_kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16}
if args.quantized:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model_kwargs["quantization_config"] = bnb_config

model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

# ----------------------------
# LoRA setup if quantized
# ----------------------------
if args.quantized:
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=True  # Fix for LLaMA 3.1 + PEFT compatibility
    )
    model = get_peft_model(model, lora_config)

# ----------------------------
# Load and format chat dataset
# ----------------------------
with open(args.dataset_path, "r", encoding="utf-8") as f:
    raw_data = [json.loads(line) for line in f if line.strip()]

def format_sample(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=True
        )
    }

dataset = Dataset.from_list([format_sample(ex) for ex in raw_data])

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=1024,
    )

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

# ----------------------------
# Training arguments
# ----------------------------
training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=2,            # was 1
    gradient_accumulation_steps=8,            # was 16
    num_train_epochs=1,                       # was 3
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    learning_rate=2e-4,
    warmup_steps=100,
    report_to="none",
    deepspeed=args.deepspeed,
    hub_model_id=args.hub_repo,
    push_to_hub=args.hub_repo is not None
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ----------------------------
# Trainer
# ----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ----------------------------
# Train and Save
# ----------------------------
trainer.train()
trainer.save_model()
tokenizer.save_pretrained(args.output_dir)