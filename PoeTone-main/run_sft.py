import os
import torch
from datasets import load_dataset

# Standard Hugging Face library imports
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ======================================================================================
#  Configuration
# ======================================================================================
SCRATCH_WORKSPACE = "out"
CACHE_DIR = os.path.join(SCRATCH_WORKSPACE, 'huggingface_cache')
os.environ['HF_HOME'] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

DATASET_FILE = os.path.join(SCRATCH_WORKSPACE, "best_of_n_dataset_Qwen.jsonl")

# --- UPDATED: Using the Instruct model is essential for chat templates ---
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

OUTPUT_DIR = os.path.join(SCRATCH_WORKSPACE, "llama3.1-8b-sft-finetuned-final")

# ======================================================================================
#  Model Loading (Standard Transformers Stack)
# ======================================================================================
print(f"Loading model: {MODEL_NAME}...")

# Configure 4-bit quantization for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the base model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    cache_dir=CACHE_DIR
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

# Manually set the chat template for Llama 3 if not set by default.
# The Instruct model is trained to understand this format.
if tokenizer.chat_template is None:
    tokenizer.chat_template = (
        "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
                "{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
            "{% elif message['role'] == 'assistant' %}"
                "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
            "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
        "{% endif %}"
    )

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

# Prepare the model for LoRA (PEFT) training
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
print("Model prepared for PEFT.")

# ======================================================================================
#  Dataset Preparation
# ======================================================================================
# Load the high-quality dataset
dataset = load_dataset("json", data_files=DATASET_FILE, split="train")

# This function prepares the dataset for the Trainer
def formatting_and_tokenizing_func(examples):
    outputs = []
    for messages in examples["messages"]:
        # Apply the chat template to format the conversation
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        outputs.append(text)
    # Tokenize the formatted strings
    return tokenizer(outputs)

tokenized_dataset = dataset.map(
    formatting_and_tokenizing_func,
    batched=True,
    remove_columns=dataset.column_names # Keep only the tokenized fields
)

print("Dataset prepared and tokenized.")

# ======================================================================================
#  Supervised Fine-Tuning with standard `transformers.Trainer`
# ======================================================================================
# Initialize the standard Hugging Face Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_dataset,
    # Data collator handles padding and creating batches
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=5e-5,
        fp16=False,
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir=OUTPUT_DIR,
    ),
)

print("\nStarting Supervised Fine-Tuning...")
trainer.train()

print("Fine-tuning complete. Saving model...")
trainer.save_model(OUTPUT_DIR)
print(f"Model saved to '{OUTPUT_DIR}'")