#!/usr/bin/env python3
"""
Simple example client for the RAG Pipeline API
Shows how to upload documents and query them
"""

import requests
import json
import sys

API_URL = "http://localhost:8000"

def check_status():
    """Check if the API is running and system status."""
    try:
        response = requests.get(f"{API_URL}/status")
        status = response.json()
        print("📊 System Status:")
        print(f"  Documents Processed: {status['documents_processed']}")
        print(f"  Vector Store Count: {status['vector_store_count']}")
        print(f"  Embedding Model: {status.get('embedding_model', 'Not loaded')}")
        return status
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to API. Is the server running?")
        print("   Start it with: uvicorn api:app --reload")
        sys.exit(1)

def upload_documents(pdf_paths, chunk_size=800, chunk_overlap=200):
    """Upload and process PDF documents."""
    print(f"\n📤 Uploading {len(pdf_paths)} document(s)...")
    
    files = []
    for pdf_path in pdf_paths:
        try:
            files.append(('files', (open(pdf_path, 'rb'))))
        except FileNotFoundError:
            print(f"❌ Error: File not found: {pdf_path}")
            return None
    
    data = {
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap
    }
    
    try:
        response = requests.post(f"{API_URL}/upload", files=files, data=data)
        
        # Close file handles
        for _, file_tuple in files:
            file_tuple[1].close()
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success!")
            print(f"  Documents Loaded: {result['documents_loaded']}")
            print(f"  Chunks Created: {result['chunks_created']}")
            print(f"  Vector Store Count: {result['vector_store_count']}")
            return result
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.json())
            return None
    except Exception as e:
        print(f"❌ Error uploading: {str(e)}")
        return None

def query(query_text, session_id=None, top_k=5, use_memory=True, metadata_filters=None):
    """Query the RAG system."""
    print(f"\n❓ Query: {query_text}")
    
    payload = {
        "query": query_text,
        "top_k": top_k,
        "use_memory": use_memory
    }
    
    if session_id:
        payload["session_id"] = session_id
    
    if metadata_filters:
        payload["metadata_filters"] = metadata_filters
    
    try:
        response = requests.post(f"{API_URL}/query", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n💡 Answer:")
            print(f"  {result['answer']}")
            print(f"\n📄 Sources ({len(result['sources'])}):")
            for i, source in enumerate(result['sources'][:3], 1):
                print(f"  {i}. Score: {source['score']:.4f}")
                print(f"     Preview: {source['preview'][:100]}...")
            print(f"\n🆔 Session ID: {result['session_id']}")
            return result
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.json())
            return None
    except Exception as e:
        print(f"❌ Error querying: {str(e)}")
        return None

def get_chat_history(session_id):
    """Get chat history for a session."""
    try:
        response = requests.get(f"{API_URL}/chat-history/{session_id}")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

if __name__ == "__main__":
    print("🚀 RAG Pipeline API Client\n")
    
    # Check status
    status = check_status()
    
    # If no documents processed, upload some
    if not status['documents_processed']:
        print("\n⚠️  No documents processed. Uploading documents...")
        pdf_paths = ["data/pdf/NIPS-2017-attention-is-all-you-need-Paper.pdf"]
        upload_result = upload_documents(pdf_paths)
        if not upload_result:
            print("❌ Failed to upload documents. Exiting.")
            sys.exit(1)
    
    # Example queries
    session_id = "example-session"
    
    print("\n" + "="*60)
    print("Example Queries")
    print("="*60)
    
    # Query 1
    result1 = query("What is attention mechanism?", session_id=session_id)
    
    # Query 2 (with memory)
    result2 = query("Who are the authors?", session_id=session_id)
    
    # Query 3 (follow-up using memory)
    result3 = query("Tell me more about it", session_id=session_id)
    
    # Show chat history
    print("\n" + "="*60)
    print("Chat History")
    print("="*60)
    history = get_chat_history(session_id)
    if history:
        print(f"Total messages: {history['message_count']}")
        for i, msg in enumerate(history['history'], 1):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')[:100]
            print(f"{i}. [{role}]: {content}...")

