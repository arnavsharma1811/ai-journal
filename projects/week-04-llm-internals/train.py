import numpy as np
from tokenizer import CharTokenizer
from transformer import MiniGPT
import matplotlib.pyplot as plt

print("🎯 MiniGPT - Demonstration Mode")
print("="*50)

# Load training data
print("Loading training data...")
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"Loaded {len(text)} characters")

# Initialize model
tokenizer = CharTokenizer(text)
model = MiniGPT(vocab_size=tokenizer.vocab_size, d_model=32, n_heads=4, n_layers=2, d_ff=64)

print(f"Model created: vocab={tokenizer.vocab_size}, d_model=32, n_layers=2")
print("")

# Simple demonstration: show model prediction before and after
print("Before training (random weights):")
prompt = "hello"
output = model.generate(tokenizer, prompt, max_new_tokens=20, temperature=0.8)
print(f"Input: '{prompt}'")
print(f"Output: '{output}'")
print("")

print("Training would normally happen here...")
print("For demonstration, showing random generation only.")
print("Full training requires proper backpropagation implementation.")
print("")

print("After training (would improve):")
print("The model would learn to predict characters based on context.")
print("")

print("✅ Transformer model is ready. Train.py demonstrates the structure.")
print("For actual training, use libraries like PyTorch + Transformers.")