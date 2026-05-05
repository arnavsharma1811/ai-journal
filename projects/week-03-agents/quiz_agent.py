import requests
import datetime
import subprocess
import chromadb
import json
import re
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS

print("🤖 AI AGENT WITH WEB SEARCH + QUIZ MODE")
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
# QUIZ STATE
# ============================================
quiz_active = False
current_question = ""
current_answer = ""
current_topic = ""
score = {"correct": 0, "total": 0}
weak_topics = {}

def add_weak_topic(topic):
    if topic in weak_topics:
        weak_topics[topic] += 1
    else:
        weak_topics[topic] = 1

def show_score():
    if score["total"] > 0:
        percent = (score["correct"] / score["total"]) * 100
        return f"Score: {score['correct']}/{score['total']} ({percent:.1f}%)"
    return "No quizzes taken yet"

# ============================================
# TOOL DESCRIPTIONS
# ============================================
tool_descriptions = """
You have access to these tools. Call them by writing TOOL: tool_name("input") on a NEW LINE.

1. search_web("query") - Use this for ANY question about future events, current news, dates, or anything not in Arnav's notes. Examples: "when is GATE 2027", "latest AI news", "what is today's date"

2. search_notes("query") - Use this ONLY for OS concepts from Arnav's notes. Examples: "FCFS scheduling", "priority inversion"

3. calculate("expression") - Use for math calculations. Example: calculate("5 + 3 * 2")

4. get_current_time() - Use ONLY when user asks for current time. Example: "what time is it"

5. run_python_code("code") - Use to run Python code

IMPORTANT: For event dates, future exams, or anything NOT in Arnav's notes, ALWAYS use search_web first.
"""

# ============================================
# TOOLS
# ============================================
def search_notes(query):
    q_emb = model.encode([query]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=3)
    if results['documents'][0]:
        return "\n\n---\n\n".join(results['documents'][0])
    return "No relevant notes found."

def search_web(query):
    """Search the web for current information"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                return "\n\n---\n\n".join([r['body'] for r in results])
            return "No results found."
    except Exception as e:
        return f"Search error: {e}"

def calculate(expression):
    """Evaluates a mathematical expression safely"""
    try:
        allowed_names = {"abs": abs, "round": round, "min": min, "max": max}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

def get_current_time():
    """Returns current date and time"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

def ask_llama(prompt):
    response = requests.post('http://localhost:11434/api/generate',
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7}
        })
    return response.json()['response']

# ============================================
# TOOL CALL PARSING
# ============================================
def extract_tool_calls(response):
    """Extract all TOOL: lines from Llama's response"""
    tool_calls = []
    lines = response.split('\n')
    for line in lines:
        if line.strip().startswith("TOOL:"):
            tool_line = line.strip()[5:].strip()
            if "(" in tool_line and ")" in tool_line:
                tool_name = tool_line.split("(")[0].strip()
                tool_input = tool_line.split("(")[1].split(")")[0].strip().strip('"').strip("'")
                
                # Skip malformed calls
                if tool_name == "calculate" and (not tool_input or tool_input == "expression"):
                    continue
                    
                tool_calls.append((tool_name, tool_input))
    return tool_calls

# ============================================
# QUIZ FUNCTIONS
# ============================================
def get_random_notes():
    """Get random notes from your database instead of top matches"""
    import random
    total_notes = collection.count()
    random_ids = [random.randint(0, total_notes - 1) for _ in range(3)]
    all_docs = collection.get()['documents']
    random_notes = []
    for idx in random_ids:
        if idx < len(all_docs):
            random_notes.append(all_docs[idx])
    return "\n\n---\n\n".join(random_notes)

def generate_question(topic=None):
    global current_question, current_answer, current_topic, quiz_active
    
    if topic:
        notes = search_notes(topic)
        prompt = f"""Based on these notes about {topic}, generate ONE GATE-style multiple choice question.

Notes: {notes[:1500]}

Output format:
QUESTION: [your question]
A) [option A]
B) [option B]  
C) [option C]
D) [option D]
ANSWER: [correct letter]
EXPLANATION: [why it's correct]

Make it challenging but fair."""
    else:
        import random
        all_docs = collection.get()['documents']
        random_notes_list = random.sample(all_docs, min(3, len(all_docs)))
        random_notes_text = "\n\n---\n\n".join(random_notes_list)
        
        prompt = f"""Based on these OS notes, generate ONE GATE-style question.

Notes: {random_notes_text[:1500]}

Output format:
QUESTION: [your question]
A) [option A]
B) [option B]  
C) [option C]
D) [option D]
ANSWER: [correct letter]
EXPLANATION: [why it's correct]"""
    
    response = ask_llama(prompt)
    
    lines = response.split('\n')
    question = ""
    options = []
    answer = ""
    explanation = ""
    
    for line in lines:
        if line.startswith("QUESTION:"):
            question = line.replace("QUESTION:", "").strip()
        elif line.startswith("A)"):
            options.append(line.strip())
        elif line.startswith("B)"):
            options.append(line.strip())
        elif line.startswith("C)"):
            options.append(line.strip())
        elif line.startswith("D)"):
            options.append(line.strip())
        elif line.startswith("ANSWER:"):
            answer = line.replace("ANSWER:", "").strip()
            match = re.search(r'[A-D]', answer.upper())
            if match:
                answer = match.group(0)
            else:
                answer = answer.rstrip(')').rstrip('.').strip().upper()
        elif line.startswith("EXPLANATION:"):
            explanation = line.replace("EXPLANATION:", "").strip()
    
    current_question = question
    current_answer = answer
    current_topic = topic if topic else "general"
    quiz_active = True
    
    print(f"\n📝 {question}")
    for opt in options:
        print(f"   {opt}")
    print(f"\n(Type your answer (A/B/C/D) or 'skip' to see answer, 'end' to stop quiz)")

def check_answer(user_answer):
    global quiz_active, current_question, current_answer, current_topic
    
    score["total"] += 1
    
    user_upper = user_answer.upper().strip()
    correct_upper = current_answer.upper().strip()
    
    if user_upper == correct_upper:
        print("\n✅ CORRECT! ✓")
        score["correct"] += 1
    else:
        print(f"\n❌ INCORRECT. Correct answer: {current_answer}")
        add_weak_topic(current_topic)
    
    print(f"   {show_score()}")
    quiz_active = False
    current_question = ""
    current_answer = ""

def show_weak_topics():
    if not weak_topics:
        print("\n📊 No weak topics identified yet. Keep taking quizzes!")
    else:
        print("\n📊 WEAK TOPICS (needs review):")
        sorted_topics = sorted(weak_topics.items(), key=lambda x: x[1], reverse=True)
        for topic, count in sorted_topics[:5]:
            print(f"   • {topic}: {count} incorrect attempts")

# ============================================
# MAIN AGENT WITH TOOLS
# ============================================
def regular_chat(user_input):
    # Step 1: Ask LLM if it needs tools
    prompt = f"""You are Arnav's GATE assistant. You have tools and his personal notes.

{tool_descriptions}

User: {user_input}

First, decide what tools you need. Output TOOL: lines for EACH tool you need, one per line.
If no tools needed, just answer directly.
"""
    
    response = ask_llama(prompt)
    
    # Step 2: Extract tool calls
    tool_calls = extract_tool_calls(response)
    
    if not tool_calls:
        # No tools needed, answer directly
        final_prompt = f"""You are Arnav's GATE assistant.
User: {user_input}
Answer directly and concisely."""
        return ask_llama(final_prompt)
    
    # Step 3: Execute tools
    print(f"\n🔧 Executing {len(tool_calls)} tools...")
    tool_results = []
    for tool_name, tool_input in tool_calls:
        print(f"   • {tool_name}('{tool_input}')")
        
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
        
        tool_results.append((tool_name, tool_input, output))
        print(f"     → {output[:100]}...")
    
    # Step 4: Build results summary
    results_text = ""
    for tool_name, tool_input, output in tool_results:
        results_text += f"\n{tool_name}('{tool_input}') returned:\n{output}\n"
    
    # Step 5: Get final answer
    final_prompt = f"""You are Arnav's GATE assistant.

User asked: {user_input}

Tool results:
{results_text}

Now provide a complete, natural answer based on the tool results above.
"""
    
    return ask_llama(final_prompt)

def run_agent(user_input):
    global quiz_active
    
    # Quiz mode commands
    if user_input.lower() in ["quiz me", "quiz", "test me"]:
        generate_question()
        return "Quiz mode activated."
    
    elif user_input.lower().startswith("quiz me on "):
        topic = user_input.lower().replace("quiz me on ", "").strip()
        generate_question(topic)
        return f"Generating quiz on {topic}..."
    
    elif user_input.lower() in ["skip", "show answer"] and quiz_active:
        print(f"\n📖 Answer: {current_answer}")
        quiz_active = False
        return "Answer shown. Type 'quiz me' for another question."
    
    elif user_input.lower() in ["end quiz", "stop quiz"] and quiz_active:
        quiz_active = False
        current_question = ""
        current_answer = ""
        return "Quiz ended. Returning to normal mode."
    
    elif quiz_active:
        match = re.search(r'[A-D]', user_input.upper())
        if match:
            check_answer(match.group(0))
            return ""
    
    elif user_input.lower() == "weak topics":
        show_weak_topics()
        return ""
    
    elif user_input.lower() == "score":
        return show_score()
    
    else:
        return regular_chat(user_input)

# ============================================
# MAIN LOOP
# ============================================
print("\n🎯 AI AGENT READY")
print("Commands:")
print("   • 'quiz me' - Random question from your notes")
print("   • 'quiz me on paging' - Question on specific topic")
print("   • 'skip' - Show answer")
print("   • 'weak topics' - Show what you're struggling with")
print("   • 'score' - Show your quiz stats")
print("   • Or just ask any question (uses web search + tools)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("\nKeep studying! 🎯")
        break
    
    response = run_agent(user_input)
    if response:
        print(f"\n🤖 {response}\n")
    print("-"*50)