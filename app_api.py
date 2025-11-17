"""
Streamlit App for RAG Pipeline using FastAPI
This app uses the REST API endpoints instead of calling functions directly.
"""

import streamlit as st
import requests
from typing import Dict, Any, Optional
import os

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="RAG Pipeline - PDF Query System (API)",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'system_status' not in st.session_state:
    st.session_state.system_status = None


def check_api_connection():
    """Check if API is accessible."""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_system_status():
    """Get system status from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/status", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None


def upload_documents_api(files, chunk_size, chunk_overlap):
    """Upload and process documents via API."""
    try:
        files_data = []
        for file in files:
            files_data.append(('files', (file.name, file.getvalue(), 'application/pdf')))
        
        data = {
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap
        }
        
        response = requests.post(
            f"{API_BASE_URL}/upload",
            files=files_data,
            data=data,
            timeout=300  # 5 minutes for large files
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            raise Exception(f"API Error: {error_detail}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Connection error: {str(e)}")


def query_api(query, session_id, top_k, use_memory, metadata_filters=None):
    """Query documents via API."""
    payload = {
        "query": query,
        "session_id": session_id,
        "top_k": top_k,
        "use_memory": use_memory
    }
    
    if metadata_filters:
        payload["metadata_filters"] = metadata_filters
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            raise Exception(f"API Error: {error_detail}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Connection error: {str(e)}")


def get_chat_history_api(session_id):
    """Get chat history from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/chat-history/{session_id}", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def clear_chat_history_api(session_id):
    """Clear chat history via API."""
    try:
        response = requests.delete(f"{API_BASE_URL}/chat-history/{session_id}", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_metadata_options():
    """Get available metadata fields from API status or query."""
    # This would require an API endpoint to get metadata fields
    # For now, we'll use a simplified approach
    status = get_system_status()
    if status and status.get('vector_store_count', 0) > 0:
        # We can't directly access vectorstore from API, so we'll use a workaround
        # Try to get metadata from a sample query or add a new API endpoint
        return {}
    return {}


def main():
    """Main Streamlit app."""
    st.title("📚 RAG Pipeline - PDF Query System (API)")
    st.markdown("Upload PDF files and query them using Retrieval-Augmented Generation (RAG) via REST API")
    
    # Check API connection
    if not check_api_connection():
        st.error(f"❌ Cannot connect to API at {API_BASE_URL}")
        st.info("Please make sure the API server is running:")
        st.code("uvicorn api:app --reload --host 0.0.0.0 --port 8000")
        st.stop()
    
    # Get system status
    if st.session_state.system_status is None:
        st.session_state.system_status = get_system_status()
    
    # Generate session ID if not exists
    if st.session_state.session_id is None:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("API Settings")
        api_url = st.text_input(
            "API Base URL",
            value=API_BASE_URL,
            help="Base URL of the FastAPI server"
        )
        if api_url != API_BASE_URL:
            st.session_state.api_base_url = api_url
            st.rerun()
        
        if st.button("🔄 Refresh Status", type="secondary"):
            st.session_state.system_status = get_system_status()
            st.rerun()
        
        st.subheader("Chunking Parameters")
        chunk_size = st.slider(
            "Chunk Size",
            min_value=200,
            max_value=2000,
            value=800,
            step=100,
            help="Size of each text chunk in characters"
        )
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=500,
            value=200,
            step=50,
            help="Number of overlapping characters between chunks"
        )
        
        st.subheader("Query Parameters")
        top_k = st.slider(
            "Top K Results",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of document chunks to retrieve"
        )
        
        st.subheader("🔍 Metadata Filters")
        st.caption("Filter documents by metadata for faster, more accurate retrieval")
        use_filters = st.checkbox("Enable Metadata Filtering", value=False)
        
        metadata_filters = {}
        if use_filters:
            st.info("💡 Metadata filtering via API - enter filter values below")
            with st.expander("Advanced: Custom Metadata Filter"):
                filter_key = st.text_input("Metadata Key", placeholder="e.g., source, page")
                filter_value = st.text_input("Metadata Value", placeholder="e.g., document.pdf or 1,2,3")
                if filter_key and filter_value:
                    # Try to parse as list if comma-separated
                    if ',' in filter_value:
                        try:
                            # Try to parse as integers
                            metadata_filters[filter_key] = [int(v.strip()) for v in filter_value.split(',')]
                        except:
                            # Keep as strings
                            metadata_filters[filter_key] = [v.strip() for v in filter_value.split(',')]
                    else:
                        try:
                            # Try integer
                            metadata_filters[filter_key] = int(filter_value)
                        except:
                            # Keep as string
                            metadata_filters[filter_key] = filter_value
        
        st.divider()
        
        st.subheader("System Status")
        status = st.session_state.system_status
        if status:
            if status.get('documents_processed'):
                st.success("✅ Documents Processed")
                st.info(f"🗄️ {status.get('vector_store_count', 0)} documents in vector store")
                if status.get('chunks_available'):
                    st.info(f"📄 {status['chunks_available']} chunks available")
            else:
                st.info("⏳ No documents processed")
            
            if status.get('embedding_model'):
                st.caption(f"Model: {status['embedding_model']}")
        else:
            st.warning("⚠️ Could not fetch system status")
        
        st.divider()
        
        if st.button("🗑️ Clear Chat History", type="secondary"):
            if st.session_state.session_id:
                if clear_chat_history_api(st.session_state.session_id):
                    st.session_state.chat_history = []
                    st.success("Chat history cleared!")
                    st.rerun()
        
        if st.button("🔄 Reset System", type="secondary"):
            try:
                response = requests.post(f"{API_BASE_URL}/reset", timeout=5)
                if response.status_code == 200:
                    st.session_state.documents_processed = False
                    st.session_state.chat_history = []
                    st.session_state.system_status = get_system_status()
                    st.success("System reset!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error resetting system: {str(e)}")
    
    # Main content area
    tab1, tab2 = st.tabs(["📤 Upload & Process", "💬 Chat"])
    
    with tab1:
        st.header("Upload PDF Files")
        st.markdown("Upload one or more PDF files to process and add to the knowledge base via API.")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="You can upload multiple PDF files at once"
        )
        
        if uploaded_files:
            st.info(f"📎 {len(uploaded_files)} file(s) selected")
            
            # Display file names
            with st.expander("View uploaded files"):
                for file in uploaded_files:
                    st.write(f"- {file.name} ({file.size:,} bytes)")
            
            if st.button("🚀 Process Documents", type="primary"):
                with st.spinner("Uploading and processing documents via API..."):
                    try:
                        result = upload_documents_api(uploaded_files, chunk_size, chunk_overlap)
                        if result.get('success'):
                            st.success("✅ Documents processed successfully!")
                            st.json(result)
                            st.session_state.documents_processed = True
                            st.session_state.system_status = get_system_status()
                        else:
                            st.error("❌ Failed to process documents.")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    with tab2:
        st.header("💬 Chat with Documents")
        st.markdown("Ask questions about the uploaded PDF documents. The chat remembers previous conversations.")
        
        # Check if documents are processed
        status = st.session_state.system_status
        if not status or not status.get('documents_processed'):
            st.warning("⚠️ Please upload and process documents first in the 'Upload & Process' tab.")
        else:
            # Display chat history
            chat_container = st.container()
            with chat_container:
                if st.session_state.chat_history:
                    for message in st.session_state.chat_history:
                        role = message.get("role", "user")
                        content = message.get("content", "")
                        
                        if role == "user":
                            with st.chat_message("user"):
                                st.write(content)
                        elif role == "assistant":
                            with st.chat_message("assistant"):
                                st.write(content)
                                # Show sources if available
                                if "sources" in message:
                                    with st.expander("📄 Sources"):
                                        for i, source in enumerate(message["sources"], 1):
                                            st.markdown(f"**Source {i}** (Score: {source.get('score', 0):.4f})")
                                            st.caption(f"Preview: {source.get('preview', '')[:200]}...")
                else:
                    st.info("👋 Start a conversation by asking a question below!")
            
            # Chat input
            query = st.chat_input(
                "Ask a question about the documents...",
                key="chat_input"
            )
            
            # Handle query
            if query:
                # Add user message to chat history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": query
                })
                
                with st.spinner("Thinking..."):
                    try:
                        # Query via API
                        result = query_api(
                            query=query,
                            session_id=st.session_state.session_id,
                            top_k=top_k,
                            use_memory=True,
                            metadata_filters=metadata_filters if use_filters and metadata_filters else None
                        )
                        
                        # Add assistant response to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result.get("sources", [])
                        })
                        
                        # Update session ID if API generated a new one
                        if result.get("session_id"):
                            st.session_state.session_id = result["session_id"]
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
                        # Remove the user message if there was an error
                        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
                            st.session_state.chat_history.pop()
            
            # Example queries
            if not st.session_state.chat_history:
                st.divider()
                st.subheader("💡 Example Queries")
                example_queries = [
                    "What is the main topic of the document?",
                    "Summarize the key points",
                    "What are the main findings?",
                ]
                
                cols = st.columns(len(example_queries))
                for i, example in enumerate(example_queries):
                    with cols[i]:
                        if st.button(f"📝 {example[:30]}...", key=f"example_{i}"):
                            st.session_state.chat_history.append({
                                "role": "user",
                                "content": example
                            })
                            st.rerun()


if __name__ == "__main__":
    main()