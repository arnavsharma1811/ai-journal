from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np

print("="*50)
print("STEP 2: Saving notes to permanent database")
print("="*50)

# 1. Load the model (same as yesterday)
print("\n📥 Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded")

# 2. Create or connect to a database on your computer
print("\n💾 Creating permanent database...")
client = chromadb.PersistentClient(path="./my_gate_notes")
print("✅ Database created at './my_gate_notes'")

# 3. Create a collection (like a folder for your notes)
try:
    collection = client.create_collection("os_notes")
    print("✅ New collection created")
except:
    collection = client.get_collection("os_notes")
    print("✅ Connected to existing collection")

# 4. Your notes (add your REAL GATE notes here)
my_notes = [
    "Paging divides memory into fixed-size frames. The MMU translates logical addresses to physical addresses using a page table.",
    "Segmentation divides memory into variable-sized segments like code, data, stack. Each segment has a base and limit.",
    "Virtual memory allows processes to run even if they're larger than physical RAM. Pages are swapped between RAM and disk.",
    "Page fault occurs when the CPU accesses a page not in memory. The OS traps, loads the page from disk, and restarts the instruction.",
    "LRU page replacement evicts the least recently used page. It's optimal but expensive to implement exactly.",
]

print(f"\n📝 Adding {len(my_notes)} notes to database...")

# 5. Generate embeddings
embeddings = model.encode(my_notes).tolist()

# 6. Save to database
ids = [f"note_{i}" for i in range(len(my_notes))]
collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=my_notes
)
print(f"✅ Saved! Database now has {collection.count()} notes")

# 7. Search function (uses saved embeddings, no recompute)
def search(query, top_k=2):
    print(f"\n🔍 Searching: '{query}'")
    query_emb = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_emb, 
        n_results=top_k
    )
    
    for i, (doc, score) in enumerate(zip(results['documents'][0], results['distances'][0])):
        similarity = 1 - score  # Chroma gives distance, convert to similarity
        print(f"\n  {i+1}. [score: {similarity:.3f}]")
        print(f"     {doc}")

# 8. Test it
print("\n" + "="*50)
print("TESTING THE DATABASE")
print("="*50)

search("How does memory management work?")
search("What happens when a page is missing?")
search("How does virtual memory let programs run?")

print("\n" + "="*50)
print("✅ Database is ready!")
print("Your notes are saved in './my_gate_notes'")
print("Close Python. Restart your computer. The notes are still there.")
print("="*50)