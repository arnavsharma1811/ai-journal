"""Simple character-level tokenizer for our mini GPT"""

class CharTokenizer:
    def __init__(self, text=None):
        if text:
            # Build vocabulary from text
            chars = sorted(list(set(text)))
            self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
            self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        else:
            # Default vocabulary (letters + space + newline)
            chars = ['a','b','c','d','e','f','g','h','i','j','k','l','m',
                     'n','o','p','q','r','s','t','u','v','w','x','y','z',
                     ' ', '\n']
            self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
            self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        self.vocab_size = len(self.char_to_idx)
    
    def encode(self, text):
        """Convert text to list of integers"""
        return [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]
    
    def decode(self, ids):
        """Convert list of integers back to text"""
        return ''.join([self.idx_to_char[i] for i in ids])

# Test
if __name__ == "__main__":
    tokenizer = CharTokenizer()
    text = "hello world"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    print(f"Original: {text}")
    print(f"Encoded: {ids}")
    print(f"Decoded: {decoded}")
    print(f"Vocab size: {tokenizer.vocab_size}")