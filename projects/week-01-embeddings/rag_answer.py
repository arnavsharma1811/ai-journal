from sentence_transformers import SentenceTransformer
import chromadb
import requests

print("="*60)
print("🚀 GATE OS ASSISTANT - RAG MODE")
print("="*60)

print("\n📥 Loading models...")

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to your database
client = chromadb.PersistentClient(path="./my_gate_notes")
collection = client.get_collection("my_notes")

print(f"✅ Loaded {collection.count()} notes")

def ask_llama(prompt):
    """Send prompt to local Llama 3"""
    try:
        response = requests.post('http://localhost:11434/api/generate',
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            },
            timeout=30
        )
        return response.json()['response']
    except Exception as e:
        return f"Error: {e}. Make sure Ollama is running (ollama serve)"

def ask_with_rag(question):
    """Retrieve relevant notes, then generate answer"""
    
    # Step 1: Find relevant notes
    q_emb = model.encode([question]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=3)
    
    # Step 2: Combine them as context
    context = "\n\n---\n\n".join(results['documents'][0])
    
    # Step 3: Build prompt for Llama
    prompt = f"""You are Arnav's GATE OS study assistant. Answer the question using ONLY the context below.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer clearly and directly
2. Use bullet points if helpful
3. If the answer is NOT in the context, say "I don't have that in my notes"
4. Be concise (2-3 paragraphs max)

ANSWER:"""
    
    # Step 4: Generate answer
    answer = ask_llama(prompt)
    return answer, results['documents'][0]

print("\n✅ Ready! Ask me anything about Operating Systems.\n")
print("-"*60)

while True:
    question = input("\n💬 You: ")
    if question.lower() in ['quit', 'exit', 'q']:
        print("\n👋 Keep studying! You got this.")
        break
    
    print("\n🤔 Searching your notes...")
    answer, sources = ask_with_rag(question)
    
    print(f"\n🤖 Answer:\n{answer}")
    
    print("\n📚 Sources used:")
    for i, src in enumerate(sources):
        preview = src[:100] + "..." if len(src) > 100 else src
        print(f"   {i+1}. {preview}")
    print("-"*60)