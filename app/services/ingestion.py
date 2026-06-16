from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os
from app.core.config import settings

def load_and_chunk_pdf(file_path : str) :

    #loading the pdf using the file path
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    #splitting the text using the recursive text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    print(f"Chopped document into {len(chunks)} chunks.")

    return chunks

def get_embedding_model() :

    # Embedding the model
    nvidia_key = os.environ.get("NVIDIA_API_KEY")
    print("Initializing NVIDIA Embedding Model...")
    embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", api_key=nvidia_key)
    
    return embeddings

def push_to_pinecone(chunks, embeddings, index_name= "sentinel-index"):

    # pushing the embeddings into the pinecone
    print(f"Uploading vectors to Pinecone index: '{index_name}'...")

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
        pinecone_api_key=settings.PINECONE_API_KEY
    )
    
if __name__ == "__main__":
    
    test_pdf_path = "data/raw/test_document.pdf" 
    
    if os.path.exists(test_pdf_path):
        document_chunks = load_and_chunk_pdf(test_pdf_path)
        nvidia_model = get_embedding_model()
        
        push_to_pinecone(document_chunks, nvidia_model)
    else:
        print(f"Error: Could not find PDF at {test_pdf_path}")