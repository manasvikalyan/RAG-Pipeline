"""
Streamlit App for RAG Pipeline with PDF Upload
This app allows users to upload PDF files, process them, and query them using RAG.
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
from typing import List

# Import RAG pipeline components
from src.rag_pipeline import (
    process_pdfs_in_directory,
    documents_chunking,
    EmbeddingModel,
    VectorStore,
    RagRetriever,
    create_groq_llm,
    rag_pipeline,
    rag_pipeline_with_memory,
    summarize_answer,
)

# Page configuration
st.set_page_config(
    page_title="RAG Pipeline - PDF Query System",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'embedding_manager' not in st.session_state:
    st.session_state.embedding_manager = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'chunked_documents' not in st.session_state:
    st.session_state.chunked_documents = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


def initialize_components():
    """Initialize RAG components if not already initialized."""
    if st.session_state.embedding_manager is None:
        with st.spinner("Loading embedding model..."):
            st.session_state.embedding_manager = EmbeddingModel()
    
    if st.session_state.llm is None:
        try:
            with st.spinner("Initializing Groq LLM..."):
                st.session_state.llm = create_groq_llm()
        except ValueError as e:
            st.error(f"Error initializing LLM: {e}")
            st.info("Please make sure GROQ_API_KEY is set in your .env file.")
            return False
    
    return True


def process_uploaded_pdfs(uploaded_files, chunk_size, chunk_overlap):
    """Process uploaded PDF files."""
    if not uploaded_files:
        st.warning("Please upload at least one PDF file.")
        return False
    
    # Create temporary directory for uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded files to temporary directory
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        # Process PDFs
        with st.spinner("Loading PDF documents..."):
            documents = process_pdfs_in_directory(temp_dir)
        
        if not documents:
            st.error("No documents were loaded. Please check your PDF files.")
            return False
        
        st.success(f"Loaded {len(documents)} document(s)")
        
        # Chunk documents
        with st.spinner(f"Chunking documents (chunk_size={chunk_size}, overlap={chunk_overlap})..."):
            chunked_documents = documents_chunking(
                documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        
        st.session_state.chunked_documents = chunked_documents
        
        # Initialize components
        if not initialize_components():
            return False
        
        # Generate embeddings
        with st.spinner("Generating embeddings..."):
            texts = [doc.page_content for doc in chunked_documents]
            embeddings = st.session_state.embedding_manager.generate_embedding(texts)
        
        # Initialize or get vector store
        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = VectorStore()
        
        # Add documents to vector store
        with st.spinner("Adding documents to vector store..."):
            st.session_state.vectorstore.add_documents(
                documents=chunked_documents,
                embeddings=embeddings
            )
        
        # Initialize or recreate retriever (to ensure it has latest method signature)
        st.session_state.retriever = RagRetriever(
            vector_store=st.session_state.vectorstore,
            embedding_manager=st.session_state.embedding_manager
        )
        
        st.session_state.documents_processed = True
        return True


def main():
    """Main Streamlit app."""
    st.title("📚 RAG Pipeline - PDF Query System")
    st.markdown("Upload PDF files and query them using Retrieval-Augmented Generation (RAG)")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
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
        if use_filters and st.session_state.vectorstore:
            try:
                # Get sample metadata to show available fields
                sample_results = st.session_state.vectorstore.collection.get(limit=1)
                if sample_results.get('metadatas') and len(sample_results['metadatas']) > 0:
                    sample_meta = sample_results['metadatas'][0]
                    available_fields = list(sample_meta.keys())
                    
                    # Filter by source file
                    if 'source' in available_fields or 'source_file' in available_fields:
                        source_field = 'source' if 'source' in available_fields else 'source_file'
                        # Get unique sources
                        all_results = st.session_state.vectorstore.collection.get()
                        unique_sources = set()
                        for meta in all_results.get('metadatas', []):
                            source = meta.get(source_field) or meta.get('source_file')
                            if source:
                                unique_sources.add(source)
                        
                        if unique_sources:
                            selected_sources = st.multiselect(
                                "Filter by Source File",
                                options=sorted(unique_sources),
                                help="Select one or more source files to search in"
                            )
                            if selected_sources:
                                # Always store as list for consistent handling
                                metadata_filters[source_field] = selected_sources if isinstance(selected_sources, list) else [selected_sources]
                    
                    # Filter by page number
                    if 'page' in available_fields:
                        page_filter = st.text_input(
                            "Filter by Page Number (optional)",
                            placeholder="e.g., 1, 2, 3 or leave empty",
                            help="Enter page numbers separated by commas"
                        )
                        if page_filter:
                            try:
                                pages = [int(p.strip()) for p in page_filter.split(',')]
                                metadata_filters['page'] = pages if len(pages) > 1 else pages[0]
                            except:
                                st.warning("Invalid page number format")
                    
                    # Custom metadata filter
                    with st.expander("Advanced: Custom Metadata Filter"):
                        filter_key = st.text_input("Metadata Key", placeholder="e.g., author, title")
                        filter_value = st.text_input("Metadata Value", placeholder="e.g., John Doe")
                        if filter_key and filter_value:
                            metadata_filters[filter_key] = filter_value
            except Exception as e:
                st.warning(f"Could not load metadata filters: {str(e)}")
        
        st.divider()
        
        st.subheader("System Status")
        if st.session_state.documents_processed:
            st.success("✅ Documents Processed")
            if st.session_state.chunked_documents:
                st.info(f"📄 {len(st.session_state.chunked_documents)} chunks available")
            if st.session_state.vectorstore:
                try:
                    count = st.session_state.vectorstore.collection.count()
                    st.info(f"🗄️ {count} documents in vector store")
                except:
                    st.warning("⚠️ Could not check vector store count")
        else:
            st.info("⏳ No documents processed")
        
        if st.button("🔄 Reset System", type="secondary"):
            st.session_state.vectorstore = None
            st.session_state.retriever = None
            st.session_state.llm = None
            st.session_state.embedding_manager = None
            st.session_state.documents_processed = False
            st.session_state.chunked_documents = None
            st.session_state.chat_history = []
            st.rerun()
        
        # Reinitialize retriever if it exists but doesn't have the new method signature
        if st.session_state.retriever and st.session_state.vectorstore and st.session_state.embedding_manager:
            import inspect
            sig = inspect.signature(st.session_state.retriever.retrieve)
            if 'metadata_filters' not in sig.parameters:
                st.session_state.retriever = RagRetriever(
                    vector_store=st.session_state.vectorstore,
                    embedding_manager=st.session_state.embedding_manager
                )
        
        st.divider()
        
        if st.button("🗑️ Clear Chat History", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main content area
    tab1, tab2 = st.tabs(["📤 Upload & Process", "💬 Chat"])
    
    with tab1:
        st.header("Upload PDF Files")
        st.markdown("Upload one or more PDF files to process and add to the knowledge base.")
        
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
                success = process_uploaded_pdfs(uploaded_files, chunk_size, chunk_overlap)
                if success:
                    st.balloons()
                    st.success("✅ Documents processed successfully! You can now chat with them in the Chat tab.")
                else:
                    st.error("❌ Failed to process documents. Please check the error messages above.")
    
    with tab2:
        st.header("💬 Chat with Documents")
        st.markdown("Ask questions about the uploaded PDF documents. The chat remembers previous conversations.")
        
        if not st.session_state.documents_processed:
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
                if st.session_state.retriever and st.session_state.llm:
                    # Add user message to chat history
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": query
                    })
                    
                    with st.spinner("Thinking..."):
                        try:
                            # First, retrieve documents to check if we have results
                            # Use score_threshold=0 to get all results, even with low similarity scores
                            # Handle both old and new method signatures
                            try:
                                results = st.session_state.retriever.retrieve(
                                    query=query,
                                    top_k=top_k,
                                    score_threshold=0,  # Get all results, even with low scores
                                    metadata_filters=metadata_filters if use_filters else None
                                )
                            except TypeError:
                                # Fallback for old method signature (without metadata_filters)
                                results = st.session_state.retriever.retrieve(
                                    query=query,
                                    top_k=top_k,
                                    score_threshold=0
                                )
                            
                            # Debug: Show retrieval info
                            if not results:
                                st.warning(f"⚠️ No documents retrieved. Vector store has {st.session_state.vectorstore.collection.count()} documents.")
                            
                            # Prepare sources for display
                            sources = [{
                                "score": r.get("score", 0),
                                "preview": r.get("document", "")[:300] + "..."
                            } for r in results] if results else []
                            
                            # Get answer using RAG pipeline with memory
                            answer = rag_pipeline_with_memory(
                                query=query,
                                retriever=st.session_state.retriever,
                                llm=st.session_state.llm,
                                conversation_history=st.session_state.chat_history[:-1],  # Exclude current query
                                top_k=top_k,
                                metadata_filters=metadata_filters if use_filters else None
                            )
                            
                            # Create concise summary for memory (store full answer in message, summary in history)
                            concise_answer = summarize_answer(answer, st.session_state.llm, max_length=150)
                            
                            # Add assistant response to chat history
                            # Store full answer for display, but concise version for memory
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": answer,  # Full answer for display
                                "concise": concise_answer,  # Concise version for memory
                                "sources": sources
                            })
                            
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error processing query: {str(e)}")
                            st.exception(e)
                            # Remove the user message if there was an error
                            if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
                                st.session_state.chat_history.pop()
                else:
                    st.error("System not properly initialized. Please process documents first.")
            
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
                            # Simulate chat input
                            st.session_state.chat_history.append({
                                "role": "user",
                                "content": example
                            })
                            st.rerun()


if __name__ == "__main__":
    main()

