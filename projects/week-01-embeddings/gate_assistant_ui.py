import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
import requests

# ============================================
# PAGE CONFIGURATION (runs first)
# ============================================
st.set_page_config(
    page_title="GATE OS Assistant",
    page_icon="📚",
    layout="wide"
)

# ============================================
# TITLE
# ============================================
st.title("📚 GATE OS Study Assistant")
st.caption("Powered by Llama 3 + RAG | Running locally on RTX 4060 | 2159 notes indexed")

# ============================================
# LOAD MODELS ONCE (cached so they don't reload)
# ============================================
@st.cache_resource
def load_models():
    """Load embedding model and connect to database (runs once)"""
    with st.spinner("Loading embedding model..."):
        model = SentenceTransformer('all-MiniLM-L6-v2')
    
    with st.spinner("Connecting to database..."):
        client = chromadb.PersistentClient(path="./my_gate_notes")
        collection = client.get_collection("my_notes")
    
    return model, collection

# Load everything
model, collection = load_models()
st.success(f"✅ Ready! {collection.count()} notes loaded")

# ============================================
# SIDEBAR SETTINGS
# ============================================
with st.sidebar:
    st.header("⚙️ Settings")
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.7)
    top_k = st.slider("Number of notes to retrieve", 1, 5, 3)
    st.divider()
    st.header("📊 Stats")
    st.metric("Total Notes", collection.count())
    st.metric("Model", "Llama 3 8B")
    st.metric("Embedding", "384 dimensions")
    st.divider()
    st.header("📚 Topics")
    st.caption("Ask about:")
    st.markdown("- Scheduling algorithms")
    st.markdown("- Process states")
    st.markdown("- Memory management")
    st.markdown("- Synchronization")
    st.markdown("- Deadlocks")

# ============================================
# FUNCTION: Ask Llama 3
# ============================================
def ask_llama(prompt, temperature):
    """Send prompt to local Llama 3"""
    try:
        response = requests.post('http://localhost:11434/api/generate',
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9
                }
            },
            timeout=30
        )
        return response.json()['response']
    except Exception as e:
        return f"⚠️ Error: {e}. Make sure Ollama is running (`ollama serve`)"

# ============================================
# FUNCTION: RAG Query
# ============================================
def rag_query(question, temperature, top_k):
    """Retrieve notes and generate answer"""
    
    # Step 1: Embed the question
    q_emb = model.encode([question]).tolist()
    
    # Step 2: Retrieve top_k notes
    results = collection.query(query_embeddings=q_emb, n_results=top_k)
    sources = results['documents'][0]
    distances = results['distances'][0]
    
    # Step 3: Build context
    context = "\n\n---\n\n".join(sources)
    
    # Step 4: Build prompt
    prompt = f"""You are Arnav's GATE OS study assistant. Answer the question using ONLY the context below.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer clearly and directly
2. Use bullet points if helpful
3. If the answer is NOT in the context, say "I don't have that in my notes"
4. Be concise

ANSWER:"""
    
    # Step 5: Generate answer
    answer = ask_llama(prompt, temperature)
    
    return answer, sources, distances

# ============================================
# CHAT INTERFACE
# ============================================

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📖 View sources"):
                for i, src in enumerate(message["sources"]):
                    similarity = 1 - message["distances"][i]
                    st.write(f"**Source {i+1}** (similarity: {similarity:.3f})")
                    st.write(src[:300] + "..." if len(src) > 300 else src)
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask me about Operating Systems..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching your notes and generating answer..."):
            answer, sources, distances = rag_query(prompt, temperature, top_k)
            st.markdown(answer)
            
            # Show sources in expander
            with st.expander("📖 View sources"):
                for i, src in enumerate(sources):
                    similarity = 1 - distances[i]
                    st.write(f"**Source {i+1}** (similarity: {similarity:.3f})")
                    preview = src[:300] + "..." if len(src) > 300 else src
                    st.write(preview)
                    if i < len(sources) - 1:
                        st.divider()
    
    # Add assistant message to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "sources": sources,
        "distances": distances
    })

# ============================================
# FOOTER
# ============================================
st.divider()
st.caption("🏃‍♂️ Built by Arnav · RTX 4060 · Local AI · 2159 notes · GATE 2027")