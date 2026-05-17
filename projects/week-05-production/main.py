from fastapi import FastAPI
from pydantic import BaseModel
import requests
import json

app = FastAPI(title="Arnav's AI Agent API")

class Query(BaseModel):
    question: str

# Your agent logic (simplified for testing)
def call_agent(question):
    # Call your local Ollama
    response = requests.post('http://localhost:11434/api/generate',
        json={
            "model": "llama3",
            "prompt": question,
            "stream": False
        })
    return response.json()['response']

@app.get("/")
def root():
    return {"message": "Arnav's AI Agent", "status": "running"}

@app.post("/ask")
def ask(query: Query):
    answer = call_agent(query.question)
    return {"question": query.question, "answer": answer}