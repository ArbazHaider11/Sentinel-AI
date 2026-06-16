import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

# --- UPDATED IMPORTS ---
from app.services.rag_pipeline import get_advanced_retriever, setup_nvidia_chain, ingest_new_pdf
from langchain_core.messages import HumanMessage, AIMessage

engine_cache = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Booting up Sentinel Engine...")
    # Using the new, lightning-fast retriever
    retriever = get_advanced_retriever()
    chain = setup_nvidia_chain(retriever)
    engine_cache["chain"] = chain
    print("✅ Sentinel is hot and ready for web traffic.")
    yield 
    print("🛑 Shutting down Sentinel API...")
    engine_cache.clear()

app = FastAPI(
    title="Sentinel Engine API", 
    description="Enterprise Hybrid-RAG backend powered by NVIDIA and Llama 3.3",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str 
    content: str

class ChatRequest(BaseModel):
    question: str
    history: Optional[List[Message]] = []

@app.get("/")
def health_check():
    return {"status": "Sentinel API is online."}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Receives a PDF from the frontend and ingests it into Pinecone."""
    try:
        os.makedirs("data/raw", exist_ok=True)
        file_path = f"data/raw/{file.filename}"
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"📥 File received: {file.filename}. Starting Pinecone ingestion...")
        chunks_processed = ingest_new_pdf(file_path)
        
        return {
            "status": "success", 
            "message": f"{file.filename} successfully ingested.",
            "chunks_embedded": chunks_processed
        }

    except Exception as e:
        print(f"❌ Upload Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def chat_with_sentinel(request: ChatRequest):
    try:
        chain = engine_cache.get("chain")
        if not chain:
            raise HTTPException(status_code=500, detail="AI Engine not loaded.")

        formatted_history = []
        for msg in request.history:
            if msg.role == "user":
                formatted_history.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                formatted_history.append(AIMessage(content=msg.content))

        print(f"Incoming web request: '{request.question}'")
        response = chain.invoke({
            "input": request.question,
            "chat_history": formatted_history
        })

        return {"answer": response["answer"]}

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")