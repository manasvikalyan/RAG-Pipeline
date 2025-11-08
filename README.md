# RAG Pipeline - Retrieval-Augmented Generation System

A comprehensive RAG (Retrieval-Augmented Generation) pipeline that processes PDF documents, creates embeddings, stores them in a vector database, and enables intelligent querying using Large Language Models with conversation memory.

## 🚀 Features

- **PDF Document Processing**: Upload and process multiple PDF files with configurable chunking
- **Semantic Search**: Vector-based document retrieval using ChromaDB for finding relevant content
- **Conversational AI**: Query documents with conversation memory that remembers previous context
- **Metadata Filtering**: Filter documents by source file, page number, or custom metadata for faster and more accurate retrieval
- **Concise Memory**: Automatically summarizes answers to keep conversation history efficient and reduce token usage
- **REST API**: Full REST API for integration with any application or custom UIs
- **Streamlit UI**: User-friendly web interface for document upload and interactive querying
- **Multiple LLM Support**: Currently supports Groq LLM (easily extensible to other providers)

## 📦 Installation

1. **Clone the repository** and navigate to the project directory
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Set up environment**: Create a `.env` file with your `GROQ_API_KEY` (get it from https://console.groq.com/)

## 🚀 Quick Start

### Streamlit Web App
Run `streamlit run app.py` and open http://localhost:8501 in your browser. Upload PDFs in the "Upload & Process" tab, then query them in the "Chat" tab.

### REST API
Run `uvicorn api:app --reload` and visit http://localhost:8000/docs for interactive API documentation. Use the API endpoints to upload documents and query them programmatically.

### Python Scripts
Use the core functions from `src/rag_pipeline.py` directly in your Python code, or run `python example_client.py` for a complete example.

## 📁 Project Structure

- **`src/rag_pipeline.py`**: Core RAG pipeline components (document processing, embeddings, vector store, retrieval, generation)
- **`app.py`**: Streamlit web application with UI for document upload and chat interface
- **`api.py`**: FastAPI REST API server for programmatic access
- **`example_client.py`**: Example Python client demonstrating API usage
- **`data/pdf/`**: Directory for PDF documents
- **`data/vector_store/`**: ChromaDB vector store persistence directory
- **`notebook/`**: Jupyter notebooks for experimentation and development

## 💡 How It Works

### Document Processing Flow

1. **Upload**: PDF files are uploaded and loaded using PyMuPDFLoader
2. **Chunking**: Documents are split into smaller chunks using RecursiveCharacterTextSplitter with configurable size and overlap
3. **Embedding**: Each chunk is converted to a vector embedding using SentenceTransformer models
4. **Storage**: Embeddings and documents are stored in ChromaDB vector database with metadata
5. **Retrieval**: When querying, the query is embedded and semantically similar documents are retrieved
6. **Generation**: Retrieved context is combined with the query and conversation history, then sent to the LLM for answer generation

### Key Components

- **EmbeddingModel**: Manages sentence transformer models for generating document and query embeddings
- **VectorStore**: Handles ChromaDB operations for storing and querying document embeddings
- **RagRetriever**: Performs semantic search with optional metadata filtering
- **RAG Pipeline Functions**: Combine retrieval with LLM generation, supporting conversation memory

### Conversation Memory

The system maintains conversation history per session, storing:
- Full user queries for context
- Concise summaries of assistant answers (extracted key points) to save space
- Previous conversation context is included in prompts to enable follow-up questions

### Metadata Filtering

Filter documents before retrieval to:
- Search only in specific source files
- Limit to certain page ranges
- Apply custom metadata filters
- Improve retrieval speed and accuracy by reducing search space

## 📚 Usage

### Streamlit App

The web interface provides two main tabs:
- **Upload & Process**: Upload PDF files, configure chunking parameters (chunk size, overlap), and process documents. View system status including document and chunk counts.
- **Chat**: Interactive chat interface where you can ask questions about uploaded documents. The chat remembers previous conversations within the session. You can enable metadata filtering in the sidebar to narrow down searches.

### REST API

The API provides endpoints for:
- **Upload**: Upload and process PDF documents with custom chunking parameters
- **Query**: Query documents with optional conversation memory and metadata filtering
- **Chat History**: Retrieve or clear conversation history for specific sessions
- **Status**: Check system status and document counts
- **Reset**: Clear all documents and chat histories

See `API_USAGE.md` for detailed API documentation and examples.

### Python Integration

Import functions from `src/rag_pipeline.py` to:
- Process PDFs from directories
- Chunk documents with custom parameters
- Generate embeddings
- Store in vector database
- Retrieve relevant documents
- Generate answers with conversation context

## ⚙️ Configuration

### Environment Variables

Set `GROQ_API_KEY` in your `.env` file to use Groq LLM models.

### Customization Options

- **Embedding Model**: Change the SentenceTransformer model (default: `all-MiniLM-L6-v2`)
- **Vector Store**: Customize collection name and persistence directory
- **LLM Model**: Choose different Groq models or extend to other providers
- **Chunking**: Adjust chunk size and overlap based on your document types
- **Retrieval**: Configure top_k results and similarity score thresholds

## 🔧 Troubleshooting

**Module not found errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`

**API key errors**: Verify your `.env` file contains the correct `GROQ_API_KEY`

**No documents retrieved**: Check that documents were successfully processed, verify the query matches document content, and try rephrasing

**Metadata filtering issues**: Ensure metadata fields exist in your documents and restart the server after code changes

**Negative similarity scores**: This is normal for some queries - the system will still return results even with low similarity

## 📖 Additional Resources

- **API Documentation**: See `API_USAGE.md` for complete REST API usage guide
- **Interactive API Docs**: Visit http://localhost:8000/docs when the API server is running
- **Example Client**: Run `python example_client.py` to see a complete usage example

## 🛠️ Technology Stack

- **LangChain**: Document processing and text splitting
- **Sentence Transformers**: Embedding generation
- **ChromaDB**: Vector database for semantic search
- **Groq**: Fast LLM inference
- **FastAPI**: REST API framework
- **Streamlit**: Web UI framework

---

**Built for efficient document querying with AI-powered retrieval and generation.**
