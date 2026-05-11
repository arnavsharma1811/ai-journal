import numpy as np

def softmax(x, axis=-1):
    """Stable softmax function"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class SingleHeadAttention:
    """Single head of self-attention"""
    def __init__(self, d_model, d_head):
        # Query, Key, Value projections
        self.W_q = np.random.randn(d_model, d_head) * 0.01
        self.W_k = np.random.randn(d_model, d_head) * 0.01
        self.W_v = np.random.randn(d_model, d_head) * 0.01
        self.d_head = d_head
    
    def forward(self, x, mask=None):
        """
        x: [batch, seq_len, d_model]
        Returns: [batch, seq_len, d_head]
        """
        # Linear projections
        Q = x @ self.W_q  # [batch, seq_len, d_head]
        K = x @ self.W_k  # [batch, seq_len, d_head]
        V = x @ self.W_v  # [batch, seq_len, d_head]
        
        # Attention scores: Q * K^T / sqrt(d_head)
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.d_head)  # [batch, seq_len, seq_len]
        
        # Apply causal mask (prevent looking at future tokens)
        if mask is None:
            # Create causal mask
            seq_len = x.shape[1]
            mask = np.triu(np.ones((seq_len, seq_len)), k=1)
            scores = scores - mask * 1e9  # Set future positions to -inf
        
        # Softmax to get attention weights
        attn_weights = softmax(scores, axis=-1)  # [batch, seq_len, seq_len]
        
        # Apply attention to values
        output = attn_weights @ V  # [batch, seq_len, d_head]
        
        return output, attn_weights

class MultiHeadAttention:
    """Multiple attention heads in parallel"""
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Create multiple heads
        self.heads = [SingleHeadAttention(d_model, self.d_head) for _ in range(n_heads)]
        
        # Output projection
        self.W_o = np.random.randn(d_model, d_model) * 0.01
    
    def forward(self, x):
        # Run each head independently
        head_outputs = []
        for head in self.heads:
            out, _ = head.forward(x)
            head_outputs.append(out)
        
        # Concatenate heads [batch, seq_len, d_model]
        concat = np.concatenate(head_outputs, axis=-1)
        
        # Final projection
        output = concat @ self.W_o
        return output

# Test
if __name__ == "__main__":
    batch_size = 2
    seq_len = 4
    d_model = 32
    n_heads = 4
    
    # Random input
    x = np.random.randn(batch_size, seq_len, d_model)
    
    # Single head
    head = SingleHeadAttention(d_model, d_model // n_heads)
    output, attn = head.forward(x, mask=True)
    print(f"Single head output shape: {output.shape}")
    print(f"Attention weights shape: {attn.shape}")
    
    # Multi-head
    mha = MultiHeadAttention(d_model, n_heads)
    output = mha.forward(x)
    print(f"Multi-head output shape: {output.shape}")