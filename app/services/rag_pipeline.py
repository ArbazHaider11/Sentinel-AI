import os
from dotenv import load_dotenv

load_dotenv()

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, NVIDIARerank, ChatNVIDIA
from langchain_pinecone import PineconeVectorStore
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- FIXING THE MISSING IMPORTS FROM INGESTION MODULE ---
from app.services.ingestion import load_and_chunk_pdf, get_embedding_model, push_to_pinecone

def ingest_new_pdf(file_path: str, index_name: str = "sentinel-index"):
    """Reads a PDF, chunks it, and pushes the vectors to Pinecone permanently."""
    print(f"Starting ingestion for: {file_path}")
    
    # 1. Read and Split
    chunks = load_and_chunk_pdf(file_path)
    
    # 2. Embed and Push to Pinecone
    embeddings = get_embedding_model()
    
    print(f"Pushing {len(chunks)} chunks to Pinecone database...")
    push_to_pinecone(chunks, embeddings, index_name)
    
    print("Ingestion complete!")
    return len(chunks)

def get_advanced_retriever(index_name="sentinel-index"):
    """Connects to Pinecone and routes results through the NVIDIA Reranker."""
    nvidia_key = os.environ.get("NVIDIA_API_KEY")
    pinecone_key = os.environ.get("PINECONE_API_KEY")

    embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", api_key=nvidia_key)
    
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=pinecone_key)
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    reranker = NVIDIARerank(
    model="nvidia/llama-nemotron-rerank-1b-v2",
    api_key=nvidia_key,
    top_n=3
    )

    advanced_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever
    )

    return advanced_retriever

def setup_nvidia_chain(retriever):
    """Sets up an active LLM endpoint for conversation."""
    nvidia_key = os.environ.get("NVIDIA_API_KEY")
    
    llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", api_key=nvidia_key)
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, formulate a standalone question. Do NOT answer it."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are Sentinel, an elite enterprise AI assistant. 
        
        INSTRUCTION 1: If the user says a basic greeting (e.g., 'hey', 'hello', 'hi', 'good morning'), politely greet them back, introduce yourself as Sentinel, and ask how you can assist them with their documents today. Do not say 'I cannot find this information' for simple greetings.
        
        INSTRUCTION 2: For all actual questions and tasks, you MUST strictly use the retrieved context below to answer. If the context does not contain the answer, state: 'I cannot find this information in the provided company documents.' Do not make up answers.
        
        Context: {context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain 