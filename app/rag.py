from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. SETUP EMBEDDINGS & LLM
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatOllama(model="llama3", temperature=0)

# 2. INGESTION FUNCTION
def ingest_document(file_path: str):
    print("Loading PDF...")
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    print("Saving to ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory="./chroma_db" 
    )
    print("Ingestion complete!")
    return vectorstore

# 3. RETRIEVAL WITH FALLBACK
def get_retriever():
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.4}
    )
    return retriever

# 4. THE RAG CHAIN
def ask_question(query: str):
    retriever = get_retriever()
    docs = retriever.invoke(query)
    
    # FALLBACK
    if not docs:
        return {"answer": "I don't have enough information in my documents to answer that.", "sources": []}
    
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))
    
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based ONLY on the following context. 
    If the answer is not in the context, say "I don't know".
    
    Context: {context}
    
    Question: {question}
    """)
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})
    
    return {"answer": answer, "sources": sources}
