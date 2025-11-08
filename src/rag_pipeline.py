"""
RAG Pipeline with Groq LLM
This module contains all the components for a Retrieval-Augmented Generation pipeline
using Groq LLM, including embedding models, vector store, retriever, and RAG functions.
"""

import os
import uuid
from typing import List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def process_pdfs_in_directory(directory_path: str):
    """
    Load all PDF files from a directory.
    
    Args:
        directory_path: Path to the directory containing PDF files
        
    Returns:
        List of Document objects loaded from PDFs
    """
    pdf_loader = DirectoryLoader(
        directory_path,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True,
    )
    documents = pdf_loader.load()
    return documents


def documents_chunking(
    documents: List[Any],
    chunk_size: int = 800,
    chunk_overlap: int = 200
) -> List[Any]:
    """
    Split documents into smaller chunks for better retrieval.
    
    Args:
        documents: List of Document objects to chunk
        chunk_size: Size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of chunked Document objects
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    # Basic cleanup before splitting - normalize whitespace
    for d in documents:
        d.page_content = " ".join(d.page_content.split())
    chunks = splitter.split_documents(documents)
    print(f"Processed {len(chunks)} chunks")
    return chunks


class EmbeddingModel:
    """Manages sentence transformer models for generating embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the sentence transformer model."""
        self.model = SentenceTransformer(self.model_name)
        print(f"Loaded model: {self.model_name}")
        print(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def generate_embedding(self, text: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            text: List of text strings to embed
            
        Returns:
            numpy array of embeddings with shape (N, D) where N is number of texts
            and D is embedding dimension
        """
        return self.model.encode(
            text,
            normalize_embeddings=True,   # ensures cosine distance in [0,2], similarity = dot
            show_progress_bar=True
        )


class VectorStore:
    """Manages ChromaDB vector store for document embeddings."""
    
    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "./data/vector_store"):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector store
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None          # chromadb.PersistentClient
        self.collection = None      # chroma Collection
        self.initialize_vector_store()

    def initialize_vector_store(self):
        """Set up ChromaDB client and collection."""
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "PDF Documents Embeddings Collection"},
        )

        print(f"Vector store initialized with collection: {self.collection_name}")
        print(f"Existing number of documents in collection: {self.collection.count()}")

    def add_documents(self, documents: List[Any], embeddings: np.ndarray, ids: Optional[List[str]] = None):
        """
        Add documents and their embeddings to the vector store.
        
        Args:
            documents: List of document objects (must have page_content and metadata attributes)
            embeddings: numpy array of embeddings with shape (N, D)
            ids: Optional list of IDs for documents. If None, UUIDs will be generated
        """
        if embeddings is None or len(documents) == 0:
            raise ValueError("documents and embeddings must be non-empty.")
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D (N, D); got shape {embeddings.shape}")
        if len(documents) != embeddings.shape[0]:
            raise ValueError("Number of documents and embeddings must be the same.")

        n = len(documents)
        ids = ids or [str(uuid.uuid4()) for _ in range(n)]
        if len(ids) != n:
            raise ValueError("Length of ids must match number of documents.")

        texts = [doc.page_content for doc in documents]
        metadatas = []
        for i, doc in enumerate(documents):
            md = dict(getattr(doc, "metadata", {}) or {})
            md["doc_index"] = i
            md["content_length"] = len(doc.page_content)
            metadatas.append(md)

        embeddings_list = embeddings.tolist()

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings_list,
        )
        print(f"Added {n} items. Collection count: {self.collection.count()}")


class RagRetriever:
    """Handles query-based retrieval from the vector store."""
    
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingModel):
        """
        Initialize the retriever.
        
        Args:
            vector_store: Vector store containing document embeddings
            embedding_manager: Manager for generating query embeddings
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self, 
        query: str, 
        top_k: int = 5, 
        score_threshold: float = 0,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query with optional metadata filtering.
        
        Args:
            query: The search query
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            metadata_filters: Optional dictionary of metadata filters (e.g., {"source": "file.pdf", "page": 1})
                            Supports filtering by any metadata field
            
        Returns:
            List of dictionaries containing retrieved documents and metadata
        """
        try:
            print(f"Retrieving top {top_k} documents for query: '{query}', score threshold: {score_threshold}")
            if metadata_filters:
                print(f"Metadata filters: {metadata_filters}")
            
            # Check if collection has documents
            collection_count = self.vector_store.collection.count()
            if collection_count == 0:
                print("Warning: Vector store collection is empty!")
                return []
            
            query_embedding = self.embedding_manager.generate_embedding([query])[0]
            
            # Build where clause for metadata filtering
            # ChromaDB requires $and operator when combining multiple conditions
            where_clause = None
            if metadata_filters and len(metadata_filters) > 0:
                conditions = []
                for key, value in metadata_filters.items():
                    # Skip None, empty dict, empty list, or empty string
                    if value is None:
                        continue
                    if isinstance(value, dict) and len(value) == 0:
                        continue
                    if isinstance(value, list):
                        # Support "in" operator for multiple values
                        if len(value) == 0:
                            continue
                        # Filter out None values from list
                        value = [v for v in value if v is not None]
                        if len(value) == 0:
                            continue
                        elif len(value) == 1:
                            # Single value in list, use direct equality
                            conditions.append({key: value[0]})
                        else:
                            # Multiple values, use $in
                            conditions.append({key: {"$in": value}})
                    elif isinstance(value, str) and len(value.strip()) == 0:
                        # Skip empty strings
                        continue
                    else:
                        # Single value (int, str, etc.) - ensure it's not empty
                        conditions.append({key: value})
                
                # ChromaDB requires exactly one operator at top level
                # If multiple conditions, wrap in $and; if single, use directly
                if len(conditions) == 0:
                    where_clause = None
                elif len(conditions) == 1:
                    where_clause = conditions[0]
                else:
                    # Multiple conditions - MUST use $and operator
                    where_clause = {"$and": conditions}
                
                if where_clause:
                    print(f"Built where clause: {where_clause}")
            
            query_params = {
                "query_embeddings": [query_embedding.tolist()],
                "n_results": min(top_k * 2, collection_count),  # Get more results to account for filtering
            }
            if where_clause:
                # Final validation: ensure where_clause has exactly one operator at top level
                # ChromaDB requires this format
                if isinstance(where_clause, dict):
                    # Check if it's already properly formatted (has $and, $or, or single key)
                    top_level_keys = list(where_clause.keys())
                    if len(top_level_keys) > 1 and '$and' not in top_level_keys and '$or' not in top_level_keys:
                        # Multiple keys without operator - this is the error case
                        # Rebuild it properly
                        conditions = [{k: v} for k, v in where_clause.items()]
                        where_clause = {"$and": conditions}
                        print(f"Fixed where clause format: {where_clause}")
                
                query_params["where"] = where_clause
                print(f"Final where clause being sent: {query_params.get('where')}")
            
            results = self.vector_store.collection.query(**query_params)
            retrieved_items = []

            if results.get('documents') and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc, md, dist, id_) in enumerate(zip(documents, metadatas, distances, ids)):
                    score = 1 - dist
                    # When score_threshold is 0, include all results (even negative scores)
                    # This handles cases where cosine similarity is negative (dissimilar documents)
                    if score_threshold == 0 or score >= score_threshold:
                        retrieved_items.append({
                            "id": id_,
                            "document": doc,
                            "metadata": md,
                            "score": score,
                            "distance": dist,
                            "rank": i + 1,
                        })
                        print(f"Retrieved doc {i}: ID={id_}, Score={score:.4f}, Distance={dist:.4f}")
                    else:
                        print(f"Doc {i} below threshold: ID={id_}, Score={score:.4f}, Distance={dist:.4f}")
            else:
                print(f"No documents retrieved. Collection has {collection_count} documents.")
            
            return retrieved_items
        except Exception as e:
            print(f"Error in retrieve: {str(e)}")
            raise


def create_groq_llm(
    model_name: str = "llama-3.1-8b-instant",
    temperature: float = 0.1,
    max_tokens: int = 1024
) -> ChatGroq:
    """
    Create and configure a Groq LLM instance.
    
    Args:
        model_name: Name of the Groq model to use
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum tokens in the response
        
    Returns:
        Configured ChatGroq instance
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in .env file.")
    
    return ChatGroq(
        model_name=model_name,
        api_key=groq_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def rag_pipeline(query: str, retriever: RagRetriever, llm: ChatGroq, top_k: int = 5) -> str:
    """
    Simple RAG pipeline that retrieves context and generates an answer.
    
    Args:
        query: User query
        retriever: RagRetriever instance
        llm: ChatGroq LLM instance
        top_k: Number of documents to retrieve
        
    Returns:
        Generated answer as a string
    """
    results = retriever.retrieve(query, top_k=top_k)
    context = "\n".join([r["document"] for r in results]) if results else ""
    if not context:
        return "No relevant information found in the context."
    
    prompt = f"""Use the following context to answer the question.
        Context: {context}
        Question: {query}
        Answer:"""
    response = llm.invoke(prompt)
    return response.content


def summarize_answer(answer: str, llm: ChatGroq, max_length: int = 150) -> str:
    """
    Extract important/key points from an answer to keep memory concise.
    
    Args:
        answer: Full answer text
        llm: LLM instance for summarization
        max_length: Maximum length of summary
        
    Returns:
        Concise summary of key points
    """
    if len(answer) <= max_length:
        return answer
    
    prompt = f"""Extract the key points and important information from this answer. 
Keep it concise (max {max_length} characters). Focus on facts, conclusions, and actionable information.

Answer:
{answer}

Key points:"""
    
    try:
        response = llm.invoke(prompt)
        summary = response.content.strip()
        return summary[:max_length] if len(summary) > max_length else summary
    except:
        # Fallback: return first part of answer
        return answer[:max_length] + "..."


def rag_pipeline_with_memory(
    query: str,
    retriever: RagRetriever,
    llm: ChatGroq,
    conversation_history: List[Dict[str, str]] = None,
    top_k: int = 5,
    metadata_filters: Optional[Dict[str, Any]] = None
) -> str:
    """
    RAG pipeline with conversation memory that includes previous context.
    
    Args:
        query: Current user query
        retriever: RagRetriever instance
        llm: ChatGroq LLM instance
        conversation_history: List of previous messages in format [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        top_k: Number of documents to retrieve
        metadata_filters: Optional dictionary of metadata filters (e.g., {"source": "file.pdf"})
        
    Returns:
        Generated answer as a string
    """
    # Retrieve with score_threshold=0 to get all results, even with low similarity
    results = retriever.retrieve(
        query, 
        top_k=top_k, 
        score_threshold=0,
        metadata_filters=metadata_filters
    )
    context = "\n".join([r["document"] for r in results]) if results else ""
    
    if not context:
        # Check if vector store has any documents
        collection_count = retriever.vector_store.collection.count()
        if collection_count == 0:
            return "No documents have been processed yet. Please upload and process documents first."
        else:
            return f"No relevant information found in the context. The vector store has {collection_count} documents, but none matched your query. Try rephrasing your question or check if the documents contain the information you're looking for."
    
    # Build conversation history string (use concise versions for memory efficiency)
    history_text = ""
    if conversation_history:
        history_text = "\n\nPrevious conversation (key points):\n"
        for msg in conversation_history[-6:]:  # Keep last 6 messages (3 exchanges)
            role = msg.get("role", "user")
            if role == "user":
                # Store full user query
                content = msg.get("content", "")
                history_text += f"User: {content}\n"
            elif role == "assistant":
                # Use concise version if available, otherwise full content
                content = msg.get("concise", msg.get("content", ""))
                history_text += f"Assistant: {content}\n"
    
    # Build prompt with context and conversation history
    prompt = f"""Use the following context from the documents to answer the question.
{history_text}
Current question: {query}

Context from documents:
{context}

Based on the context above and the conversation history, provide a helpful answer. If the question refers to something from the previous conversation, use that context along with the document context.
Answer:"""
    
    response = llm.invoke(prompt)
    return response.content


# def rag_advanced(
#     query: str,
#     retriever: RagRetriever,
#     llm: ChatGroq,
#     top_k: int = 5,
#     min_score: float = 0.2,
#     return_context: bool = False
# ) -> Dict[str, Any]:
#     """
#     Advanced RAG pipeline with extra features:
#     - Returns answer, sources, confidence score, and optionally full context.
    
#     Args:
#         query: User query
#         retriever: RagRetriever instance
#         llm: ChatGroq LLM instance
#         top_k: Number of documents to retrieve
#         min_score: Minimum similarity score threshold
#         return_context: Whether to include full context in the response
        
#     Returns:
#         Dictionary containing:
#         - answer: Generated answer
#         - sources: List of source documents with metadata
#         - confidence: Maximum similarity score
#         - context: Full context (if return_context=True)
#     """
#     results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)
#     if not results:
#         return {
#             'answer': 'No relevant context found.',
#             'sources': [],
#             'confidence': 0.0,
#             'context': ''
#         }
    
#     # Prepare context and sources
#     context = "\n\n".join([doc['document'] for doc in results])
#     sources = [{
#         'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
#         'page': doc['metadata'].get('page', 'unknown'),
#         'score': doc['score'],
#         'preview': doc['document'][:300] + '...'
#     } for doc in results]
#     confidence = max([doc['score'] for doc in results])
    
#     # Generate answer
#     prompt = f"""Use the following context to answer the question concisely.
# Context:
# {context}

# Question: {query}

# Answer:"""
#     response = llm.invoke(prompt)
    
#     output = {
#         'answer': response.content,
#         'sources': sources,
#         'confidence': confidence
#     }
#     if return_context:
#         output['context'] = context
#     return output


# Example usage (commented out)
"""
if __name__ == "__main__":
    # Step 1: Load PDF documents
    documents = process_pdfs_in_directory("../data/pdf")
    
    # Step 2: Chunk documents
    chunked_documents = documents_chunking(documents, chunk_size=800, chunk_overlap=200)
    
    # Step 3: Initialize components
    embedding_manager = EmbeddingModel()
    vectorstore = VectorStore()
    
    # Step 4: Generate embeddings and add to vector store
    texts = [doc.page_content for doc in chunked_documents]
    embeddings = embedding_manager.generate_embedding(texts)
    vectorstore.add_documents(documents=chunked_documents, embeddings=embeddings)
    
    # Step 5: Initialize retriever and LLM
    retriever = RagRetriever(vector_store=vectorstore, embedding_manager=embedding_manager)
    llm = create_groq_llm()
    
    # Step 6: Use RAG pipeline
    answer = rag_pipeline("What is attention?", retriever, llm)
    print(answer)
"""

