import requests
import datetime
import subprocess
import chromadb
from sentence_transformers import SentenceTransformer

print("🧠 AGENT WITH MEMORY (Remembers everything)")
print("="*60)

# ============================================
# LOAD YOUR RAG SYSTEM
# ============================================
print("Loading your notes database...")
model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="../week-01-embeddings/my_gate_notes")
collection = client.get_collection("my_notes")
print(f"✅ Loaded {collection.count()} notes")

# ============================================
# MEMORY - NEW
# ============================================
conversation_memory = []  # Stores all Q&A

def add_to_memory(role, content):
    """Add a message to conversation memory"""
    conversation_memory.append({"role": role, "content": content})
    # Keep only last 10 exchanges (prevents context overflow)
    if len(conversation_memory) > 20:
        conversation_memory.pop(0)

def get_conversation_context():
    """Get recent conversation history as string"""
    if not conversation_memory:
        return ""
    context = "Previous conversation:\n"
    for msg in conversation_memory[-6:]:  # Last 3 exchanges
        role = "User" if msg["role"] == "user" else "Assistant"
        context += f"{role}: {msg['content']}\n"
    return context

# ============================================
# TOOLS (same as before)
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
    q_emb = model.encode([query]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=2)
    if results['documents'][0]:
        return "\n\n---\n\n".join(results['documents'][0])
    return "No relevant notes found."

# ============================================
# TOOL DESCRIPTIONS (updated with memory context)
# ============================================
tool_descriptions = """
You have access to these tools. Call them by writing TOOL: tool_name("input") on a NEW LINE.

1. search_notes("query") - Search Arnav's GATE OS notes. Use for any OS concept question.
2. calculate("expression") - For math. Example: calculate("5 + 3 * 2")
3. get_current_time() - For current time/date
4. run_python_code("code") - To run Python code

IMPORTANT: If you need MULTIPLE tools, output each on a separate line.
"""

# ============================================
# OLLAMA CALL
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
# PARSE MULTIPLE TOOLS
# ============================================
def extract_tool_calls(response):
    tool_calls = []
    lines = response.split('\n')
    for line in lines:
        if line.strip().startswith("TOOL:"):
            tool_line = line.strip()[5:].strip()
            if "(" in tool_line and ")" in tool_line:
                tool_name = tool_line.split("(")[0].strip()
                tool_input = tool_line.split("(")[1].split(")")[0].strip().strip('"').strip("'")
                tool_calls.append((tool_name, tool_input))
    return tool_calls

def execute_tool(tool_name, tool_input):
    if tool_name == "search_notes":
        return search_notes(tool_input)
    elif tool_name == "get_current_time":
        return get_current_time()
    elif tool_name == "calculate":
        return calculate(tool_input)
    elif tool_name == "run_python_code":
        return run_python_code(tool_input)
    else:
        return f"Unknown tool: {tool_name}"

# ============================================
# MULTI-TOOL AGENT WITH MEMORY
# ============================================
def run_agent(user_input):
    # Step 1: Get conversation history
    conversation_context = get_conversation_context()
    
    # Step 2: Build prompt with memory
    prompt = f"""You are Arnav's GATE assistant. You have tools and his personal notes.

{tool_descriptions}

{conversation_context}

User: {user_input}

First, decide what tools you need. Output TOOL: lines for EACH tool you need, one per line.
If no tools needed, just answer directly using the conversation context if relevant.
"""
    
    response = ask_llama(prompt)
    
    # Step 3: Extract tool calls
    tool_calls = extract_tool_calls(response)
    
    if not tool_calls:
        # No tools needed, answer directly
        final_answer = ask_llama(f"{conversation_context}\nUser: {user_input}\nAnswer directly using conversation context if relevant.")
        # Add to memory
        add_to_memory("user", user_input)
        add_to_memory("assistant", final_answer)
        return final_answer, []
    
    # Step 4: Execute tools
    print(f"\n🔧 Executing {len(tool_calls)} tools...")
    tool_results = []
    for tool_name, tool_input in tool_calls:
        print(f"   • {tool_name}('{tool_input}')")
        output = execute_tool(tool_name, tool_input)
        tool_results.append((tool_name, tool_input, output))
        print(f"     → {output[:100]}...")
    
    # Step 5: Build results summary
    results_text = ""
    for tool_name, tool_input, output in tool_results:
        results_text += f"\n{tool_name}('{tool_input}') returned:\n{output}\n"
    
    # Step 6: Get final answer with memory context
    final_prompt = f"""You are Arnav's GATE assistant.

{conversation_context}

User asked: {user_input}

Tool results:
{results_text}

Now provide a complete, natural answer to the user combining ALL the information above.
If the user is asking for a comparison (e.g., "compare with previous"), use the conversation context.
"""
    
    final_response = ask_llama(final_prompt)
    
    # Step 7: Add to memory
    add_to_memory("user", user_input)
    add_to_memory("assistant", final_response)
    
    return final_response, tool_results

# ============================================
# MAIN LOOP
# ============================================
print("\n✅ AGENT WITH MEMORY READY")
print("I remember our conversation. Ask follow-up questions like 'compare with previous'\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    
    print("\n🤔 Thinking...")
    response, tools_used = run_agent(user_input)
    
    if tools_used:
        print(f"\n🔧 Tools used: {len(tools_used)}")
        for tool_name, tool_input, _ in tools_used:
            print(f"   • {tool_name}('{tool_input}')")
    
    print(f"\n🤖 Assistant:\n{response}\n")
    print("-"*60)