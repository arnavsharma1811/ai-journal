from sentence_transformers import SentenceTransformer
import chromadb

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Connecting to database...")
client = chromadb.PersistentClient(path="./my_gate_notes")


collection = client.get_collection("my_notes")

print(f"✅ Connected! Found {collection.count()} notes")
print("\n" + "="*60)
print("🔥 YOUR GATE OS KNOWLEDGE BASE IS LIVE")
print(f"📚 Searching across {collection.count()} notes")
print("="*60 + "\n")

while True:
    query = input("Ask about OS (or 'quit'): ")
    if query.lower() in ['quit', 'exit', 'q']:
        break
    
    print("🔍 Searching...")
    q_emb = model.encode([query]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=3)
    
    print(f"\n📖 Top matches from your notes:\n")
    for i, doc in enumerate(results['documents'][0]):
        similarity = 1 - results['distances'][0][i]
        print(f"{i+1}. [Score: {similarity:.3f}]")
        display_text = doc[:300] + "..." if len(doc) > 300 else doc
        print(f"   {display_text}")
        print()
    print("-" * 60)