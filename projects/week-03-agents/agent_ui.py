import streamlit as st
import requests
import datetime
import subprocess
import chromadb
import re
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS

st.set_page_config(page_title="Arnav's AI Agent", page_icon="🤖", layout="wide")
st.title("🤖 Arnav's AI Agent")
st.caption("RAG + Web Search + Tools | Llama 3 on RTX 4060")

# ============================================
# LOAD MODELS (cached)
# ============================================
@st.cache_resource
def load_rag():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="../week-01-embeddings/my_gate_notes")
    collection = client.get_collection("my_notes")
    return model, collection

model, collection = load_rag()
st.success(f"✅ {collection.count()} notes loaded | Ollama: Llama 3")

# ============================================
# TOOLS (same as quiz_agent.py)
# ============================================
def search_notes(query):
    q_emb = model.encode([query]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=3)
    if results['documents'][0]:
        return "\n\n---\n\n".join(results['documents'][0])
    return "No relevant notes found."

def search_web(query):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                return "\n\n---\n\n".join([r['body'] for r in results])
            return "No results found."
    except Exception as e:
        return f"Search error: {e}"

def calculate(expression):
    try:
        allowed_names = {"abs": abs, "round": round, "min": min, "max": max}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except:
        return f"Error: Invalid expression"

def get_current_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def run_python_code(code):
    try:
        result = subprocess.run(["python", "-c", code], capture_output=True, text=True, timeout=5)
        return result.stdout if result.stdout else result.stderr or "Code executed"
    except:
        return "Error: Timeout"

def ask_llama(prompt):
    response = requests.post('http://localhost:11434/api/generate',
        json={"model": "llama3", "prompt": prompt, "stream": False, "options": {"temperature": 0.7}})
    return response.json()['response']

def extract_tool_calls(response):
    tool_calls = []
    for line in response.split('\n'):
        if line.strip().startswith("TOOL:"):
            tool_line = line.strip()[5:].strip()
            if "(" in tool_line and ")" in tool_line:
                tool_name = tool_line.split("(")[0].strip()
                tool_input = tool_line.split("(")[1].split(")")[0].strip().strip('"').strip("'")
                tool_calls.append((tool_name, tool_input))
    return tool_calls

tool_descriptions = """
1. search_web("query") - For future events, news, dates
2. search_notes("query") - For OS concepts from notes
3. calculate("expression") - For math
4. get_current_time() - For current time
5. run_python_code("code") - To run Python code
"""

def run_agent(user_input):
    prompt = f"""You are Arnav's assistant. Tools:
{tool_descriptions}

User: {user_input}

If you need a tool, output TOOL: tool_name("input") on a new line.
"""
    response = ask_llama(prompt)
    tool_calls = extract_tool_calls(response)
    
    if not tool_calls:
        final_prompt = f"User: {user_input}\nAnswer directly."
        return ask_llama(final_prompt), []
    
    results_text = ""
    for tool_name, tool_input in tool_calls:
        if tool_name == "search_notes":
            output = search_notes(tool_input)
        elif tool_name == "search_web":
            output = search_web(tool_input)
        elif tool_name == "calculate":
            output = calculate(tool_input)
        elif tool_name == "get_current_time":
            output = get_current_time()
        elif tool_name == "run_python_code":
            output = run_python_code(tool_input)
        else:
            output = f"Unknown tool: {tool_name}"
        results_text += f"\n{tool_name}('{tool_input}'):\n{output}\n"
    
    final_prompt = f"User asked: {user_input}\n\nTool results:\n{results_text}\n\nProvide final answer."
    return ask_llama(final_prompt), tool_calls

# ============================================
# CHAT UI
# ============================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "tools" in msg and msg["tools"]:
            with st.expander("🔧 Tools used"):
                for tool in msg["tools"]:
                    st.code(f"{tool[0]}(\"{tool[1]}\")")

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, tools_used = run_agent(prompt)
            st.markdown(response)
            if tools_used:
                with st.expander("🔧 Tools used"):
                    for tool in tools_used:
                        st.code(f"{tool[0]}(\"{tool[1]}\")")
    
    st.session_state.messages.append({"role": "assistant", "content": response, "tools": tools_used})