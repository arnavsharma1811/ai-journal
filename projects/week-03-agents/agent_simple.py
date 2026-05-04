import requests
import json
import datetime
import subprocess

print("🤖 GATE Assistant Agent (Tool-Using AI)")
print("="*50)

# ============================================
# TOOLS
# ============================================

def get_current_time():
    """Returns current date and time"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate(expression):
    """Evaluates a mathematical expression safely"""
    try:
        # Safe evaluation (only math operations)
        allowed_names = {"abs": abs, "round": round, "min": min, "max": max}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

def run_python_code(code):
    """Executes Python code and returns output"""
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

# ============================================
# TOOL DESCRIPTION FOR LLM
# ============================================

tool_descriptions = """
You have access to these tools. Call them by writing TOOL: tool_name("input")

1. get_current_time() - Use when user asks for current time or date
2. calculate("expression") - Use for math calculations. Example: calculate("5 + 3 * 2")
3. run_python_code("code") - Use to run Python code. Example: run_python_code("print(sum(range(10)))")

If no tool is needed, answer directly.
"""

# ============================================
# CALL OLLAMA
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
# AGENT FUNCTION
# ============================================

def run_agent(user_input):
    # Step 1: Ask LLM if it needs a tool
    prompt = f"""You are an AI assistant with tool access.

{tool_descriptions}

User: {user_input}

Think step by step. If you need a tool, output exactly:
TOOL: tool_name("input")

If no tool needed, answer directly.
"""
    
    response = ask_llama(prompt)
    
    # Step 2: Check if response contains a tool call
    if "TOOL:" in response:
        # Extract tool call
        tool_line = response.split("TOOL:")[-1].split("\n")[0].strip()
        
        if "(" in tool_line and ")" in tool_line:
            tool_name = tool_line.split("(")[0].strip()
            tool_input = tool_line.split("(")[1].split(")")[0].strip().strip('"').strip("'")
            
            print(f"\n🔧 Using tool: {tool_name}('{tool_input}')")
            
            # Step 3: Execute tool
            if tool_name == "get_current_time":
                output = get_current_time()
            elif tool_name == "calculate":
                output = calculate(tool_input)
            elif tool_name == "run_python_code":
                output = run_python_code(tool_input)
            else:
                output = f"Unknown tool: {tool_name}"
            
            print(f"📤 Tool output: {output}\n")
            
            # Step 4: Get final answer from LLM with tool output
            final_prompt = f"""User asked: {user_input}

A tool was called: {tool_name}("{tool_input}")
Tool returned: {output}

Now provide the final answer to the user based on this tool output.
"""
            final_response = ask_llama(final_prompt)
            return final_response
    
    return response

# ============================================
# MAIN LOOP
# ============================================

print("Tools available: time, calculate, run Python code\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    
    print("\n🤔 Thinking...")
    response = run_agent(user_input)
    print(f"\n🤖 Assistant: {response}\n")
    print("-"*50)