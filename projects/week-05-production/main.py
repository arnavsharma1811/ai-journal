from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json

app = FastAPI(title="Arnav's AI Agent API")

class Query(BaseModel):
    question: str

def call_agent(question):
    # Demo mode: return a structured response without calling LLM
    if "fcfs" in question.lower():
        return "FCFS (First-Come-First-Served) is a non-preemptive scheduling algorithm where processes are executed in the order they arrive."
    elif "priority" in question.lower():
        return "Priority inversion occurs when a high-priority task is forced to wait for a low-priority task holding a shared resource."
    else:
        return f"You asked: {question}. This is a demo response. The full agent will be available soon."

@app.get("/")
def root():
    return {"message": "Arnav's AI Agent API", "status": "running"}

@app.post("/ask")
def ask(query: Query):
    try:
        answer = call_agent(query.question)
        return {"question": query.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))