import requests
import datetime
import subprocess
import chromadb
from sentence_transformers import SentenceTransformer

print("🤖 GATE Agent + RAG (Notes Search + Tools)")
print("="*60)

# ============================================
# 1. LOAD YOUR RAG SYSTEM (Week 1)
# ============================================
print("Loading your notes database...")
model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="../week-01-embeddings/my_gate_notes")
collection = client.get_collection("my_notes")
print(f"✅ Loaded {collection.count()} notes")

# ============================================
# 2. TOOLS (from agent_simple.py)
# ============================================
def get_current_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate(expression):
    try:
        allowed_names = {"abs": abs, "round": round, "min": min, "max": max}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

def run_python_code(code):
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.stdout:
            return result.stdout
        elif result.stderr:
            return f"Error: {result.stderr}"
        return "Code executed (no output)"
    except subprocess.TimeoutExpired:
        return "Error: Code timed out"

def search_notes(query):
    """Search your personal GATE notes using semantic search"""
    q_emb = model.encode([query]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=2)
    if results['documents'][0]:
        return "\n\n---\n\n".join(results['documents'][0])
    return "No relevant notes found."

# ============================================
# 3. TOOL DESCRIPTIONS FOR LLM
# ============================================
tool_descriptions = """
You have access to these tools. Call them by writing TOOL: tool_name("input")

1. search_notes("query") - Search Arnav's GATE OS notes. Use for any OS concept question.
2. calculate("expression") - For math. Example: calculate("5 + 3 * 2")
3. get_current_time() - For current time/date
4. run_python_code("code") - To run Python code

If no tool needed, answer directly.
"""

# ============================================
# 4. OLLAMA CALL
# ============================================
def ask_llama(prompt):
    response = requests.post('http://localhost:11434/api/generate',
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3}
        })
    return response.json()['response']

# ============================================
# 5. AGENT LOOP (Same as before, + search_notes)
# ============================================
def run_agent(user_input):
    prompt = f"""You are Arnav's GATE assistant. You have tools and his personal notes.

{tool_descriptions}

User: {user_input}

If you need a tool, output: TOOL: tool_name("input")
If no tool needed, answer directly.
"""
    
    response = ask_llama(prompt)
    
    if "TOOL:" in response:
        tool_line = response.split("TOOL:")[-1].split("\n")[0].strip()
        
        if "(" in tool_line and ")" in tool_line:
            tool_name = tool_line.split("(")[0].strip()
            tool_input = tool_line.split("(")[1].split(")")[0].strip().strip('"').strip("'")
            
            print(f"\n🔧 Using tool: {tool_name}('{tool_input}')")
            
            if tool_name == "search_notes":
                output = search_notes(tool_input)
            elif tool_name == "get_current_time":
                output = get_current_time()
            elif tool_name == "calculate":
                output = calculate(tool_input)
            elif tool_name == "run_python_code":
                output = run_python_code(tool_input)
            else:
                output = f"Unknown tool: {tool_name}"
            
            print(f"📤 Tool output: {output[:200]}...")
            
            final_prompt = f"""User asked: {user_input}
Tool called: {tool_name}("{tool_input}")
Tool returned: {output}
Now provide final answer to the user based on this tool output."""
            
            final_response = ask_llama(final_prompt)
            return final_response
    
    return response

# ============================================
# 6. MAIN LOOP
# ============================================
print("\n✅ Agent + RAG ready. Ask me anything about OS or use tools.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    
    print("\n🤔 Thinking...")
    response = run_agent(user_input)
    print(f"\n🤖 Assistant: {response}\n")
    print("-"*60)