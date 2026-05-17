import requests
import json
import time

print("="*60)
print("RAG EVALUATION WITH LOCAL LLAMA 3")
print("="*60)

def ask_llama(prompt):
    """Query your local Llama 3"""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0}
        },
        timeout=30
    )
    return response.json()['response']

def evaluate_faithfulness(question, context, answer):
    """Does the answer stay true to the context?"""
    prompt = f"""You are an evaluator. Rate faithfulness from 1-10.

Context: {context}
Question: {question}
Answer: {answer}

Does the answer use ONLY information from the context? No outside knowledge?

Respond with ONLY a number 1-10. 10 = completely faithful.
"""
    response = ask_llama(prompt)
    try:
        return int(response.strip())
    except:
        return 5

def evaluate_relevancy(question, answer):
    """Does the answer directly address the question?"""
    prompt = f"""You are an evaluator. Rate relevancy from 1-10.

Question: {question}
Answer: {answer}

Does the answer directly address what was asked? Ignore correctness.

Respond with ONLY a number 1-10. 10 = perfectly relevant.
"""
    response = ask_llama(prompt)
    try:
        return int(response.strip())
    except:
        return 5

# Test questions
test_cases = [
    {
        "question": "What is FCFS scheduling?",
        "context": "FCFS (First-Come-First-Served) is a non-preemptive CPU scheduling algorithm where processes execute in arrival order.",
        "your_answer": "FCFS is First-Come-First-Served. Processes run in the order they arrive. No preemption."
    },
    {
        "question": "What is priority inversion?",
        "context": "Priority inversion occurs when a high-priority task waits for a low-priority task holding a shared resource.",
        "your_answer": "Priority inversion happens when a high-priority process is blocked by a low-priority process holding a lock."
    }
]

results = []

for i, test in enumerate(test_cases, 1):
    print(f"\n📝 Test {i}: {test['question']}")
    print("-"*40)
    
    faithfulness = evaluate_faithfulness(test['question'], test['context'], test['your_answer'])
    relevancy = evaluate_relevancy(test['question'], test['your_answer'])
    
    print(f"🎯 Faithfulness: {faithfulness}/10")
    print(f"📌 Relevancy: {relevancy}/10")
    
    results.append({
        "question": test['question'],
        "faithfulness": faithfulness,
        "relevancy": relevancy
    })
    
    time.sleep(1)  # Small delay to avoid overwhelming Ollama

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
total_faith = sum(r['faithfulness'] for r in results) / len(results)
total_rel = sum(r['relevancy'] for r in results) / len(results)

print(f"Average Faithfulness: {total_faith:.1f}/10")
print(f"Average Relevancy: {total_rel:.1f}/10")