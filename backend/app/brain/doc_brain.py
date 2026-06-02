import time
import uuid
import re
import os
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec

from app.config import GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME
from app.database import db

# Configure Gemini — also set GOOGLE_API_KEY env var for embed_content() compatibility
if GEMINI_API_KEY and not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

class DocBrain:
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = PINECONE_INDEX_NAME
        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)

    def _ensure_index_exists(self):
        """Ensures that the Pinecone index exists with appropriate dimensions for text-embedding-004 (768)."""
        try:
            available_indexes = [idx.name for idx in self.pc.list_indexes()]
            if self.index_name not in available_indexes:
                print(f"Pinecone index '{self.index_name}' not found. Creating it serverless...")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=3072,  # Google gemini-embedding-2 size
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                # Wait for index to initialize
                while not self.pc.describe_index(self.index_name).status['ready']:
                    time.sleep(2)
                print(f"Pinecone index '{self.index_name}' successfully initialized!")
            else:
                print(f"Pinecone index '{self.index_name}' is ready.")
        except Exception as e:
            print(f"Warning during Pinecone index initialization check: {e}")

    def semantic_chunk(self, text: str, max_chunk_size: int = 700, overlap: int = 150) -> List[str]:
        """
        Splits text by sentence boundaries, trying to pack complete sentences 
        into semantic blocks under max_chunk_size.
        """
        # Clean spacing
        text = re.sub(r'\s+', ' ', text).strip()
        # Split by sentence endings (., !, ?) keeping the separator
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_chunk_size:
                current_chunk += (" " + sentence if current_chunk else sentence)
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                if len(current_chunk) > overlap:
                    overlap_text = current_chunk[-overlap:]
                    # Try to align overlap to start of a sentence in the overlap area
                    word_boundary = overlap_text.find(" ")
                    if word_boundary != -1:
                        overlap_text = overlap_text[word_boundary:]
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
                    
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        """Generates embedding using models/gemini-embedding-2 from Google."""
        task_type = "retrieval_query" if is_query else "retrieval_document"
        try:
            response = genai.embed_content(
                model="models/gemini-embedding-2",
                content=text,
                task_type=task_type
            )
            return response["embedding"]
        except Exception as e:
            print(f"Failed to generate Gemini embedding: {e}")
            # Return dummy vector for zero-crash fallback
            return [0.0] * 3072

    def ingest_document(self, title: str, content: str, source_type: str = "url", source_url: Optional[str] = None) -> Dict[str, Any]:
        """Chunks a document, embeds the chunks, and uploads them to the corresponding namespace in Pinecone."""
        chunks = self.semantic_chunk(content)
        vectors = []
        
        doc_id = str(uuid.uuid4())
        source = source_url or f"{source_type}://{title.replace(' ', '_').lower()}"
        
        print(f"Ingesting '{title}' into namespace '{source_type}' ({len(chunks)} chunks)...")
        
        for idx, chunk in enumerate(chunks):
            embedding = self.get_embedding(chunk, is_query=False)
            chunk_id = f"{doc_id}_chunk_{idx}"
            
            metadata = {
                "doc_id": doc_id,
                "title": title,
                "source": source,
                "source_type": source_type,
                "text": chunk,
                "chunk_index": idx,
                "timestamp": str(time.time())
            }
            
            vectors.append({
                "id": chunk_id,
                "values": embedding,
                "metadata": metadata
            })
            
            # Save to SQLite local database for fallback and premium book library
            db.save_knowledge_chunk(
                doc_id=doc_id,
                chunk_index=idx,
                title=title,
                source=source,
                source_type=source_type,
                text=chunk
            )
            
        # Batch upsert to Pinecone
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            try:
                self.index.upsert(vectors=batch, namespace=source_type)
            except Exception as e:
                print(f"Error upserting vectors to Pinecone: {e}")
                
        # Log to local SQLite DB as system action
        db.log_event(
            source="DocBrain",
            message=f"Ingested document: '{title}' ({len(chunks)} chunks)",
            status="success",
            meta_dict={"doc_id": doc_id, "title": title, "source_type": source_type, "chunks": len(chunks)}
        )
        
        return {
            "doc_id": doc_id,
            "title": title,
            "chunks_count": len(chunks),
            "source_type": source_type,
            "source": source
        }

    def query_knowledge(self, query: str, namespaces: List[str] = None, limit: int = 4) -> List[Dict[str, Any]]:
        """Queries the Pinecone index across multiple namespaces and returns matches with confidence scores."""
        if not namespaces:
            namespaces = ["url", "notion", "pdf", "gmail", "whatsapp"]

        if not GEMINI_API_KEY:
            print("[DocBrain] GEMINI_API_KEY missing. Using local SQLite keyword search for RAG context.")
            return db.search_knowledge_chunks(query, limit)
            
        query_vector = self.get_embedding(query, is_query=True)
        if not any(query_vector):
            print("[DocBrain] Embedding returned a zero vector. Using local SQLite keyword search for RAG context.")
            return db.search_knowledge_chunks(query, limit)

        results = []
        
        for ns in namespaces:
            try:
                response = self.index.query(
                    namespace=ns,
                    vector=query_vector,
                    top_k=limit,
                    include_metadata=True
                )
                
                for match in response.get("matches", []):
                    # Map cosine similarity (-1.0 to 1.0) into a clean confidence percentage (0 to 100)
                    score = match.get("score", 0.0)
                    confidence = max(0.0, min(100.0, (score + 1.0) / 2.0 * 100.0))
                    
                    metadata = match.get("metadata", {})
                    results.append({
                        "doc_id": metadata.get("doc_id"),
                        "title": metadata.get("title", "Untitled Document"),
                        "source": metadata.get("source"),
                        "source_type": metadata.get("source_type", ns),
                        "text": metadata.get("text", ""),
                        "chunk_index": metadata.get("chunk_index", 0),
                        "confidence": round(confidence, 1),
                        "score": score
                    })
            except Exception as e:
                print(f"Pinecone query error in namespace '{ns}': {e}")
                
        # Fallback to local SQLite search if Pinecone yields zero matches
        if not results:
            print("[DocBrain] Pinecone RAG returned no matches. Invoking local SQLite fallback keyword search...")
            results = db.search_knowledge_chunks(query, limit)
            
        # Sort combined results by confidence/score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

# Global doc brain instance
doc_brain = DocBrain()
