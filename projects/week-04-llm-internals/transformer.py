import numpy as np
from attention import MultiHeadAttention

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class FeedForward:
    """Simple 2-layer neural network"""
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)
    
    def forward(self, x):
        hidden = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        output = hidden @ self.W2 + self.b2
        return output

class TransformerBlock:
    """One transformer block: Attention + FeedForward with residuals"""
    def __init__(self, d_model, n_heads, d_ff):
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        
        # Layer normalization parameters
        self.ln1_scale = np.ones(d_model)
        self.ln1_shift = np.zeros(d_model)
        self.ln2_scale = np.ones(d_model)
        self.ln2_shift = np.zeros(d_model)
    
    def layer_norm(self, x, scale, shift, eps=1e-5):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(var + eps)
        return normalized * scale + shift
    
    def forward(self, x):
        # Self-attention with residual
        attn_out = self.attention.forward(x)
        x = self.layer_norm(x + attn_out, self.ln1_scale, self.ln1_shift)
        
        # Feed-forward with residual
        ff_out = self.feed_forward.forward(x)
        x = self.layer_norm(x + ff_out, self.ln2_scale, self.ln2_shift)
        
        return x

class MiniGPT:
    """Complete mini GPT model"""
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=4, d_ff=128, max_seq_len=100):
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Embeddings
        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.01
        self.position_embedding = np.random.randn(max_seq_len, d_model) * 0.01
        
        # Transformer blocks
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        
        # Output layer
        self.output_linear = np.random.randn(d_model, vocab_size) * 0.01
        self.output_bias = np.zeros(vocab_size)
    
    def forward(self, idx):
        batch, seq_len = idx.shape
        
        # Token embeddings
        token_emb = self.token_embedding[idx]
        
        # Position embeddings
        pos_emb = self.position_embedding[:seq_len, :]
        pos_emb = np.expand_dims(pos_emb, axis=0)
        
        # Add embeddings
        x = token_emb + pos_emb
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x)
        
        # Output projection
        logits = x @ self.output_linear + self.output_bias
        
        return logits
    
    def generate(self, tokenizer, prompt, max_new_tokens=50, temperature=1.0):
        idx = tokenizer.encode(prompt)
        
        for _ in range(max_new_tokens):
            context = idx[-100:]
            context = np.array([context])
            
            logits = self.forward(context)
            last_logits = logits[0, -1, :] / temperature
            
            exp_logits = np.exp(last_logits - np.max(last_logits))
            probs = exp_logits / np.sum(exp_logits)
            
            next_token = np.random.choice(len(probs), p=probs)
            idx.append(next_token)
        
        return tokenizer.decode(idx)

# Test
if __name__ == "__main__":
    from tokenizer import CharTokenizer
    
    tokenizer = CharTokenizer()
    model = MiniGPT(vocab_size=tokenizer.vocab_size, d_model=32, n_heads=4, n_layers=2, d_ff=64)
    
    print("Testing forward pass...")
    dummy_input = np.array([[0,1,2,3,4,5]])
    logits = model.forward(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Vocabulary size: {logits.shape[-1]}")
    print("✅ Transformer model ready!")