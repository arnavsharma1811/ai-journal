import requests
import datetime
import subprocess
import chromadb
import re
from sentence_transformers import SentenceTransformer

print("🤖 MULTI-TOOL AGENT (Handles multiple tools at once)")
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
# TOOLS
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
# TOOL DESCRIPTIONS
# ============================================
tool_descriptions = """
You have access to these tools. Call them by writing TOOL: tool_name("input") on a NEW LINE.

1. search_notes("query") - Search Arnav's GATE OS notes. Use for any OS concept question.
2. calculate("expression") - For math. Example: calculate("5 + 3 * 2")
3. get_current_time() - For current time/date
4. run_python_code("code") - To run Python code

IMPORTANT: If you need MULTIPLE tools, output each on a separate line:
TOOL: search_notes("FCFS scheduling")
TOOL: calculate("0.15 * 900")

Then after all tools, I'll give you the results and you provide the final answer.
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
    """Extract all TOOL: lines from Llama's response"""
    tool_calls = []
    lines = response.split('\n')
    for line in lines:
        if line.strip().startswith("TOOL:"):
            tool_line = line.strip()[5:].strip()  # Remove "TOOL:"
            if "(" in tool_line and ")" in tool_line:
                tool_name = tool_line.split("(")[0].strip()
                tool_input = tool_line.split("(")[1].split(")")[0].strip().strip('"').strip("'")
                tool_calls.append((tool_name, tool_input))
    return tool_calls

# ============================================
# EXECUTE TOOL
# ============================================
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
# MULTI-TOOL AGENT
# ============================================
def run_agent(user_input):
    # Step 1: Ask Llama what tools it needs
    prompt = f"""You are Arnav's GATE assistant. You have tools and his personal notes.

{tool_descriptions}

User: {user_input}

First, decide what tools you need. Output TOOL: lines for EACH tool you need, one per line.
Then after the tools, say "I need these tools to answer the question."

Example:
TOOL: search_notes("FCFS scheduling")
TOOL: calculate("0.15 * 900")
I need these tools to answer the question.

Do NOT answer the user yet. Just output the tool calls.
"""
    
    response = ask_llama(prompt)
    print(f"\n📝 Llama wants tools: {response[:200]}...")
    
    # Step 2: Extract all tool calls
    tool_calls = extract_tool_calls(response)
    
    if not tool_calls:
        # No tools needed, answer directly
        final_prompt = f"""User: {user_input}
Answer directly."""
        return ask_llama(final_prompt)
    
    # Step 3: Execute each tool
    print(f"\n🔧 Executing {len(tool_calls)} tools...")
    tool_results = []
    for tool_name, tool_input in tool_calls:
        print(f"   • {tool_name}('{tool_input}')")
        output = execute_tool(tool_name, tool_input)
        tool_results.append((tool_name, tool_input, output))
        print(f"     → {output[:100]}...")
    
    # Step 4: Build results summary
    results_text = ""
    for tool_name, tool_input, output in tool_results:
        results_text += f"\n{tool_name}('{tool_input}') returned:\n{output}\n"
    
    # Step 5: Get final answer
    final_prompt = f"""User asked: {user_input}

Tool results:
{results_text}

Now provide a complete, natural answer to the user combining ALL the information above.
If multiple tools were used, combine their results into one response.
Be specific. Use the exact numbers and facts from the tool outputs.
"""
    
    final_response = ask_llama(final_prompt)
    return final_response, tool_results

# ============================================
# MAIN LOOP
# ============================================
print("\n✅ MULTI-TOOL AGENT READY")
print("Ask me things that need multiple tools (e.g., 'What is FCFS and calculate 15% of 900')\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    
    print("\n🤔 Thinking...")
    response, tools_used = run_agent(user_input)
    
    print(f"\n🔧 Tools used: {len(tools_used)}")
    for tool_name, tool_input, _ in tools_used:
        print(f"   • {tool_name}('{tool_input}')")
    
    print(f"\n🤖 Assistant:\n{response}\n")
    print("-"*60)