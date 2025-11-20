"""
RAG Pipeline REST API
A FastAPI-based REST API for the RAG Pipeline system.
Can be used from terminal, other applications, or to build custom UIs.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import tempfile
import uuid
from pathlib import Path

# Import RAG pipeline components
from src.rag_pipeline import (
    process_pdfs_in_directory,
    documents_chunking,
    EmbeddingModel,
    VectorStore,
    RagRetriever,
    create_groq_llm,
    rag_pipeline_with_memory,
    summarize_answer,
)

app = FastAPI(
    title="RAG Pipeline API",
    description="REST API for Retrieval-Augmented Generation with PDF documents",
    version="1.0.0"
)

# Enable CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (in production, use a proper state management system)
global_state = {
    "vectorstore": None,
    "retriever": None,
    "llm": None,
    "embedding_manager": None,
    "documents_processed": False,
    "chat_histories": {}  # Store chat histories per session
}


# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    top_k: int = 5
    metadata_filters: Optional[Dict[str, Any]] = None
    use_memory: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    session_id: str
    message: str


class ProcessDocumentsRequest(BaseModel):
    chunk_size: int = 800
    chunk_overlap: int = 200
    collection_name: Optional[str] = None
    persist_directory: Optional[str] = None


class ProcessDocumentsResponse(BaseModel):
    success: bool
    message: str
    documents_loaded: int
    chunks_created: int
    vector_store_count: int


class SystemStatusResponse(BaseModel):
    documents_processed: bool
    vector_store_count: int
    chunks_available: Optional[int]
    embedding_model: Optional[str]


class ChatHistoryResponse(BaseModel):
    session_id: str
    history: List[Dict[str, str]]
    message_count: int


def initialize_components():
    """Initialize RAG components if not already initialized."""
    if global_state["embedding_manager"] is None:
        global_state["embedding_manager"] = EmbeddingModel()
    
    if global_state["llm"] is None:
        try:
            global_state["llm"] = create_groq_llm()
        except ValueError as e:
            raise HTTPException(status_code=500, detail=f"Error initializing LLM: {str(e)}")


@app.get("/")
async def root():
    """API root endpoint with information."""
    return {
        "message": "RAG Pipeline API",
        "version": "1.0.0",
        "endpoints": {
            "POST /upload": "Upload and process PDF documents",
            "POST /query": "Query documents using RAG",
            "GET /status": "Get system status",
            "GET /chat-history/{session_id}": "Get chat history for a session",
            "DELETE /chat-history/{session_id}": "Clear chat history for a session",
            "POST /reset": "Reset the entire system",
            "GET /docs": "API documentation (Swagger UI)"
        }
    }


@app.get("/health")
async def health_check():
    """Simple health endpoint to verify that the API server is running."""
    return {"status": "ok"}


@app.get("/status", response_model=SystemStatusResponse)
async def get_status():
    """Get the current status of the RAG system."""
    chunks_available = None
    if global_state.get("chunked_documents"):
        chunks_available = len(global_state["chunked_documents"])
    
    vector_store_count = 0
    if global_state["vectorstore"]:
        try:
            vector_store_count = global_state["vectorstore"].collection.count()
        except:
            pass
    
    embedding_model = None
    if global_state["embedding_manager"]:
        embedding_model = global_state["embedding_manager"].model_name
    
    return SystemStatusResponse(
        documents_processed=global_state["documents_processed"],
        vector_store_count=vector_store_count,
        chunks_available=chunks_available,
        embedding_model=embedding_model
    )


@app.post("/upload", response_model=ProcessDocumentsResponse)
async def upload_and_process_documents(
    files: List[UploadFile] = File(...),
    chunk_size: int = Form(800),
    chunk_overlap: int = Form(200),
    collection_name: Optional[str] = Form(None),
    persist_directory: Optional[str] = Form(None)
):
    """
    Upload PDF files and process them for RAG.
    
    - **files**: List of PDF files to upload
    - **chunk_size**: Size of text chunks (default: 800)
    - **chunk_overlap**: Overlap between chunks (default: 200)
    - **collection_name**: Optional custom collection name
    - **persist_directory**: Optional custom persist directory
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Create temporary directory for uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded files
        for file in files:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
            
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
        
        try:
            # Process PDFs
            documents = process_pdfs_in_directory(temp_dir)
            if not documents:
                raise HTTPException(status_code=400, detail="No documents were loaded from PDFs")
            
            documents_count = len(documents)
            
            # Chunk documents
            chunked_documents = documents_chunking(
                documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            global_state["chunked_documents"] = chunked_documents
            
            # Initialize components
            initialize_components()
            
            # Generate embeddings
            texts = [doc.page_content for doc in chunked_documents]
            embeddings = global_state["embedding_manager"].generate_embedding(texts)
            
            # Initialize or get vector store
            if global_state["vectorstore"] is None:
                global_state["vectorstore"] = VectorStore(
                    collection_name=collection_name or "pdf_documents",
                    persist_directory=persist_directory or "./data/vector_store"
                )
            
            # Add documents to vector store
            global_state["vectorstore"].add_documents(
                documents=chunked_documents,
                embeddings=embeddings
            )
            
            # Initialize retriever
            global_state["retriever"] = RagRetriever(
                vector_store=global_state["vectorstore"],
                embedding_manager=global_state["embedding_manager"]
            )
            
            global_state["documents_processed"] = True
            vector_store_count = global_state["vectorstore"].collection.count()
            
            return ProcessDocumentsResponse(
                success=True,
                message="Documents processed successfully",
                documents_loaded=documents_count,
                chunks_created=len(chunked_documents),
                vector_store_count=vector_store_count
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query documents using RAG with optional conversation memory.
    
    - **query**: The question to ask
    - **session_id**: Optional session ID for conversation memory (auto-generated if not provided)
    - **top_k**: Number of documents to retrieve (default: 5)
    - **metadata_filters**: Optional metadata filters (e.g., {"source": "file.pdf", "page": 1})
    - **use_memory**: Whether to use conversation history (default: True)
    """
    if not global_state["documents_processed"]:
        raise HTTPException(
            status_code=400,
            detail="No documents processed. Please upload and process documents first using /upload endpoint."
        )
    
    if not global_state["retriever"] or not global_state["llm"]:
        raise HTTPException(
            status_code=500,
            detail="System not properly initialized. Please process documents first."
        )
    
    # Generate or use session ID
    session_id = request.session_id or str(uuid.uuid4())
    
    # Get or create chat history for this session
    if session_id not in global_state["chat_histories"]:
        global_state["chat_histories"][session_id] = []
    
    chat_history = global_state["chat_histories"][session_id]
    
    try:
        # Clean and validate metadata filters
        cleaned_filters = None
        if request.metadata_filters:
            cleaned_filters = {}
            for key, value in request.metadata_filters.items():
                # Skip empty values, None, empty dicts, empty lists, empty strings
                if value is None:
                    continue
                if isinstance(value, dict) and len(value) == 0:
                    continue
                if isinstance(value, list) and len(value) == 0:
                    continue
                if isinstance(value, str) and len(value.strip()) == 0:
                    continue
                # Only add valid filters
                cleaned_filters[key] = value
            
            # If all filters were invalid, set to None
            if len(cleaned_filters) == 0:
                cleaned_filters = None
        
        # Retrieve documents
        results = global_state["retriever"].retrieve(
            query=request.query,
            top_k=request.top_k,
            score_threshold=0,
            metadata_filters=cleaned_filters
        )
        
        # Prepare sources
        sources = [{
            "score": r.get("score", 0),
            "preview": r.get("document", "")[:300] + "...",
            "metadata": r.get("metadata", {}),
            "id": r.get("id", "")
        } for r in results] if results else []
        
        # Get answer using RAG pipeline
        conversation_history = chat_history if request.use_memory else None
        answer = rag_pipeline_with_memory(
            query=request.query,
            retriever=global_state["retriever"],
            llm=global_state["llm"],
            conversation_history=conversation_history,
            top_k=request.top_k,
            metadata_filters=request.metadata_filters
        )
        
        # Create concise summary for memory
        concise_answer = summarize_answer(answer, global_state["llm"], max_length=150)
        
        # Update chat history
        chat_history.append({
            "role": "user",
            "content": request.query
        })
        chat_history.append({
            "role": "assistant",
            "content": answer,
            "concise": concise_answer,
            "sources": sources
        })
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            session_id=session_id,
            message="Query processed successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/chat-history/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str):
    """Get chat history for a specific session."""
    if session_id not in global_state["chat_histories"]:
        raise HTTPException(status_code=404, detail="Session not found")
    
    history = global_state["chat_histories"][session_id]
    return ChatHistoryResponse(
        session_id=session_id,
        history=history,
        message_count=len(history)
    )


@app.delete("/chat-history/{session_id}")
async def clear_chat_history(session_id: str):
    """Clear chat history for a specific session."""
    if session_id in global_state["chat_histories"]:
        global_state["chat_histories"][session_id] = []
        return {"message": f"Chat history cleared for session {session_id}"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@app.post("/reset")
async def reset_system():
    """Reset the entire RAG system (clears all documents and chat histories)."""
    global_state["vectorstore"] = None
    global_state["retriever"] = None
    global_state["llm"] = None
    global_state["embedding_manager"] = None
    global_state["documents_processed"] = False
    global_state["chunked_documents"] = None
    global_state["chat_histories"] = {}
    
    return {"message": "System reset successfully"}


@app.get("/sessions")
async def list_sessions():
    """List all active chat sessions."""
    return {
        "sessions": list(global_state["chat_histories"].keys()),
        "count": len(global_state["chat_histories"])
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

