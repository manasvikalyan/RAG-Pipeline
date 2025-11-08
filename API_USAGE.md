# RAG Pipeline API Usage Guide

This API provides a REST interface to the RAG Pipeline system, allowing you to use it from the terminal, build custom UIs, or integrate it into other applications.

## Starting the API Server

```bash
# Using uvicorn directly
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Or using Python
python api.py
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Endpoints

### 1. Get API Information
```bash
curl http://localhost:8000/
```

### 2. Check System Status
```bash
curl http://localhost:8000/status
```

### 3. Upload and Process PDF Documents

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "files=@/path/to/document1.pdf" \
  -F "files=@/path/to/document2.pdf" \
  -F "chunk_size=800" \
  -F "chunk_overlap=200"
```

**Parameters:**
- `files`: PDF files to upload (can upload multiple)
- `chunk_size`: Size of text chunks (default: 800)
- `chunk_overlap`: Overlap between chunks (default: 200)
- `collection_name`: Optional custom collection name
- `persist_directory`: Optional custom persist directory

### 4. Query Documents

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is attention mechanism?",
    "top_k": 5,
    "use_memory": true
  }'
```

**With session ID (for conversation memory):**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Who are the authors?",
    "session_id": "my-session-123",
    "top_k": 5,
    "use_memory": true
  }'
```

**With metadata filters:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is attention?",
    "top_k": 5,
    "metadata_filters": {
      "source": ["../data/pdf/NIPS-2017-attention-is-all-you-need-Paper.pdf"],
      "page": 1
    }
  }'
```

**Response:**
```json
{
  "answer": "The answer from the RAG system...",
  "sources": [
    {
      "score": 0.85,
      "preview": "Document preview...",
      "metadata": {...},
      "id": "doc-id"
    }
  ],
  "session_id": "auto-generated-or-provided",
  "message": "Query processed successfully"
}
```

### 5. Get Chat History

```bash
curl http://localhost:8000/chat-history/{session_id}
```

### 6. Clear Chat History

```bash
curl -X DELETE http://localhost:8000/chat-history/{session_id}
```

### 7. List All Sessions

```bash
curl http://localhost:8000/sessions
```

### 8. Reset System

```bash
curl -X POST http://localhost:8000/reset
```

## Python Client Example

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000"

# 1. Upload documents
with open("document.pdf", "rb") as f:
    files = {"files": f}
    data = {"chunk_size": 800, "chunk_overlap": 200}
    response = requests.post(f"{BASE_URL}/upload", files=files, data=data)
    print(response.json())

# 2. Query documents
query_data = {
    "query": "What is attention mechanism?",
    "session_id": "my-session",
    "top_k": 5,
    "use_memory": True
}
response = requests.post(f"{BASE_URL}/query", json=query_data)
result = response.json()
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")

# 3. Continue conversation
query_data = {
    "query": "Tell me more about it",
    "session_id": "my-session",  # Same session ID
    "top_k": 5,
    "use_memory": True
}
response = requests.post(f"{BASE_URL}/query", json=query_data)
print(response.json()["answer"])

# 4. Get chat history
response = requests.get(f"{BASE_URL}/chat-history/my-session")
print(response.json())
```

## JavaScript/TypeScript Example

```javascript
// Upload documents
const formData = new FormData();
formData.append('files', fileInput.files[0]);
formData.append('chunk_size', '800');
formData.append('chunk_overlap', '200');

const uploadResponse = await fetch('http://localhost:8000/upload', {
  method: 'POST',
  body: formData
});
const uploadResult = await uploadResponse.json();
console.log(uploadResult);

// Query documents
const queryResponse = await fetch('http://localhost:8000/query', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    query: 'What is attention mechanism?',
    session_id: 'my-session',
    top_k: 5,
    use_memory: true
  })
});
const queryResult = await queryResponse.json();
console.log(queryResult.answer);
```

## Building a Custom Streamlit App

You can use the API from your own Streamlit app:

```python
import streamlit as st
import requests

API_URL = "http://localhost:8000"

# Query function
def query_rag(query, session_id=None):
    response = requests.post(
        f"{API_URL}/query",
        json={
            "query": query,
            "session_id": session_id,
            "top_k": 5,
            "use_memory": True
        }
    )
    return response.json()

# Use in your Streamlit app
st.title("My Custom RAG App")
query = st.text_input("Ask a question")
if query:
    result = query_rag(query, session_id="my-session")
    st.write(result["answer"])
```

## Features

✅ **Document Upload & Processing**: Upload PDFs and process them into chunks
✅ **RAG Querying**: Query documents with retrieval-augmented generation
✅ **Conversation Memory**: Maintain conversation history per session
✅ **Metadata Filtering**: Filter documents by source, page, or custom metadata
✅ **Concise Memory**: Automatically summarizes answers for efficient memory storage
✅ **Session Management**: Multiple concurrent chat sessions
✅ **RESTful API**: Standard REST endpoints for easy integration

## Error Handling

All endpoints return appropriate HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid input)
- `404`: Not Found (session/resource not found)
- `500`: Internal Server Error

Error responses include a `detail` field with the error message.

