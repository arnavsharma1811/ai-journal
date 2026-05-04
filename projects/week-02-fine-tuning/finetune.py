from unsloth import FastLanguageModel
import torch
from datasets import Dataset
import json

print("🚀 Starting fine-tuning...")
print("="*60)

# 1. Load base model in 4-bit (fits on 8GB VRAM)
print("\n📥 Loading Llama 3 (4-bit quantization)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-2-3b-instruct-bnb-4bit",
    max_seq_length=2048,  # Context window
    dtype=None,  # Auto-detect
    load_in_4bit=True,  # 4-bit quantization = fits on 8GB GPU
)

print("✅ Model loaded")

# 2. Add LoRA adapters (efficient fine-tuning)
print("\n🔧 Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank (higher = more capacity)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

print("✅ LoRA adapters added")

# 3. Load training data
print("\n📚 Loading training data...")
with open("training_data.json", "r", encoding="utf-8") as f:
    training_data = json.load(f)

# Format as Alpaca-style prompts
def format_prompt(example):
    return f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}"""

formatted_data = [format_prompt(item) for item in training_data]
dataset = Dataset.from_dict({"text": formatted_data})

print(f"✅ Loaded {len(dataset)} examples")

# 4. Tokenize dataset
print("\n🔤 Tokenizing dataset...")
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
print("✅ Tokenization complete")

# 5. Train
print("\n🏋️ Starting training (this will take 30-60 minutes)...")
print("   Your RTX 4060 is about to work hard.")
print("   Go grab a coffee or stretch.\n")

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./gate_llama_finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,  # Mixed precision training
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    remove_unused_columns=False,
    report_to="none",  # Disable wandb
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

# 6. Save the fine-tuned model
print("\n💾 Saving fine-tuned model...")
model.save_pretrained("gate_llama_finetuned")
tokenizer.save_pretrained("gate_llama_finetuned")

print("\n" + "="*60)
print("✅ FINE-TUNING COMPLETE!")
print("📍 Model saved to: ./gate_llama_finetuned")
print("="*60)