from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import ask_question

app = FastAPI(title="Solvity Local RAG API")

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"status": "Local API is running"}

@app.post("/ask")
def ask(request: QueryRequest):
    result = ask_question(request.question)
    return result
