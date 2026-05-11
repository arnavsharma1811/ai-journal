import numpy as np

class TokenEmbedding:
    """Converts token IDs to vectors"""
    def __init__(self, vocab_size, d_model):
        # Randomly initialize embedding matrix [vocab_size x d_model]
        self.weights = np.random.randn(vocab_size, d_model) * 0.01
        self.d_model = d_model
    
    def forward(self, x):
        """x: [batch, seq_len] → output: [batch, seq_len, d_model]"""
        return self.weights[x]

class PositionalEncoding:
    """Adds information about token position in sequence"""
    def __init__(self, max_seq_len, d_model):
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        # Pre-compute positional encodings
        self.positions = np.zeros((max_seq_len, d_model))
        
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                # Sin for even dimensions
                self.positions[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
                if i + 1 < d_model:
                    # Cos for odd dimensions
                    self.positions[pos, i + 1] = np.cos(pos / (10000 ** ((i + 1) / d_model)))
    
    def forward(self, x):
        """x: [batch, seq_len, d_model] → add position info"""
        seq_len = x.shape[1]
        return x + self.positions[:seq_len, :]

# Test
if __name__ == "__main__":
    vocab_size = 27
    d_model = 32
    batch_size = 2
    seq_len = 5
    
    # Create embeddings
    token_emb = TokenEmbedding(vocab_size, d_model)
    pos_emb = PositionalEncoding(max_seq_len=100, d_model=d_model)
    
    # Sample token IDs
    tokens = np.array([[0,1,2,3,4], [5,6,7,8,9]])
    print(f"Input tokens shape: {tokens.shape}")
    
    # Get embeddings
    embedded = token_emb.forward(tokens)
    print(f"Token embeddings shape: {embedded.shape}")
    
    # Add positional encoding
    output = pos_emb.forward(embedded)
    print(f"Final embeddings shape: {output.shape}")