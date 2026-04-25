from sentence_transformers import SentenceTransformer
import numpy as np


print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!")

# Some test sentences
sentences = [
    "Operating systems manage computer hardware",
    "Deadlock occurs when processes wait for each other",
    "Python is a programming language",
    "Machine learning helps computers learn from data",
    "Pizza is a delicious food"
]

# Convert sentences to embeddings (numbers)
print("\nConverting sentences to embeddings...")
embeddings = model.encode(sentences)

# Check what we got
print(f"\nEach sentence is now {len(embeddings[0])} numbers!")
print(f"First sentence as numbers: {embeddings[0][:5]}...")  # Show first 5

# Function to find similar sentences
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Test it: Which sentence is most similar to a query?
query = "What is an operating system?"
query_embedding = model.encode(query)

print(f"\nQuery: '{query}'")
print("\nSimilarity scores:")

for i, sentence in enumerate(sentences):
    similarity = cosine_similarity(query_embedding, embeddings[i])
    print(f"{similarity:.3f} - {sentence}")