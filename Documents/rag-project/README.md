# Solvity Local RAG Assistant

A completely local, zero-cost Retrieval-Augmented Generation (RAG) pipeline. It is designed to answer questions based strictly on internal company documentation while actively preventing hallucinations.

## 🏗️ Architecture
- **Ingestion:** `PyPDFLoader` + `RecursiveCharacterTextSplitter` (500 character chunks, 50 overlap).
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (runs entirely on-device).
- **Vector Database:** ChromaDB (persisted locally).
- **LLM:** Llama 3 via Ollama (runs entirely on-device).
- **API:** FastAPI with automatic Swagger UI documentation.

## 🛡️ The Hallucination Guardrail (Key Feature)
To prevent the LLM from answering questions outside the scope of the provided documents, I implemented a **similarity score threshold fallback** at the retrieval layer.

If the top retrieved chunk scores below `0.4` against the user's query, the system intercepts the request, skips the LLM entirely, and returns: `"I don't have enough information in my documents to answer that."`

This ensures the system fails safely instead of confidently guessing.

## 🚀 How to Run Locally
1. Ensure [Ollama](https://ollama.com/) is running locally and you have pulled the Llama 3 model (`ollama pull llama3`).
2. Activate your virtual environment and install dependencies: 
   ```bash
   pip install -r requirements.txt
   