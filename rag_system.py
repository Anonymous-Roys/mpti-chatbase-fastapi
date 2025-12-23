import json
import os
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

EMBEDDINGS_MODEL = 'all-MiniLM-L6-v2'
CHUNKS_FILE = "processed_chunks.json"
EMBEDDINGS_FILE = "embeddings.json"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class RAGSystem:
    def __init__(self, website_name: str = "the website"):
        """Initializes the RAG System, loads data, and sets up models."""
        self.website_name = website_name
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray = np.array([])
        self.encoder: Optional[SentenceTransformer] = None
        self.llm_client: Optional[Groq] = None
        self.is_ready = False
        
        self.load_models()
        self.load_or_create_embeddings()
        self.is_ready = self.embeddings.size > 0
        
        if not GROQ_API_KEY:
             print("WARNING: GROQ_API_KEY not found. LLM generation will fail.")

    def load_models(self):
        """Loads the Sentence Transformer and Groq client."""
        try:
            print(f"Loading Sentence Transformer model: {EMBEDDINGS_MODEL}...")
            self.encoder = SentenceTransformer(EMBEDDINGS_MODEL)
            print("Encoder loaded successfully.")
        except Exception as e:
            print(f"Error loading Sentence Transformer: {e}")
            self.encoder = None

        if GROQ_API_KEY:
            try:
                self.llm_client = Groq(api_key=GROQ_API_KEY)
                print("Groq client initialized.")
            except Exception as e:
                print(f"Error initializing Groq client: {e}")
                self.llm_client = None

    def load_or_create_embeddings(self):
        """Loads chunks and embeddings, creating embeddings if necessary."""
        self.chunks = self._load_chunks()

        if os.path.exists(EMBEDDINGS_FILE):
            print(f"Loading pre-computed embeddings from {EMBEDDINGS_FILE}...")
            try:
                with open(EMBEDDINGS_FILE, 'r') as f:
                    embeddings_list = json.load(f)
                    self.embeddings = np.array(embeddings_list, dtype=np.float32)
                print(f"Loaded {len(self.embeddings)} embeddings.")
                return
            except Exception as e:
                print(f"Error loading embeddings: {e}. Re-creating embeddings.")

        if self.chunks and self.encoder:
            print(f"Creating embeddings for {len(self.chunks)} chunks...")
            # Extract text content for encoding
            texts = [chunk["text"] for chunk in self.chunks]
            # Encode texts
            embeddings_list = self.encoder.encode(texts, convert_to_numpy=True).tolist()
            self.embeddings = np.array(embeddings_list, dtype=np.float32)

            # Save embeddings for future use
            with open(EMBEDDINGS_FILE, 'w') as f:
                json.dump(embeddings_list, f)
            print(f"Embeddings created and saved to {EMBEDDINGS_FILE}.")
        elif not self.chunks:
            print("No processed chunks found. Run data_processor.py first.")
        
    def _load_chunks(self) -> List[Dict[str, Any]]:
        """Utility to load processed chunks."""
        if not os.path.exists(CHUNKS_FILE):
            return []
        with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)

    def retrieve_context(self, query: str, top_k: int = 3) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Retrieves the top_k relevant chunks using cosine similarity.
        Returns (list of unique source URLs, list of relevant chunk objects)
        """
        if not self.is_ready or not self.encoder:
            return [], []

        # 1. Encode the query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)[0]

        # 2. Calculate cosine similarity (dot product of normalized vectors)
        # Normalize the stored embeddings (already done by the model, but for safety)
        # self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1)[:, np.newaxis] 
        # Normalize the query embedding
        # query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Cosine similarity is simply the dot product if vectors are unit length
        # Since sentence-transformers already ensures unit vectors, we can use dot product
        similarities = np.dot(self.embeddings, query_embedding)

        # 3. Get top_k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # 4. Extract chunks and sources
        relevant_chunks = [self.chunks[i] for i in top_indices]
        sources = list(set(chunk["source_url"] for chunk in relevant_chunks))

        return sources, relevant_chunks

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generates an answer using the Groq LLM based on provided context."""
        if not self.llm_client:
            return "LLM service is not configured (missing GROQ_API_KEY or failed initialization)."

        context_text = "\n---\n".join([chunk["text"] for chunk in context_chunks])
        
        prompt = f"""
You are a helpful assistant for {self.website_name}. Answer based ONLY on the context provided.
If the answer is not in the context, say: "I don't have that information. Please check the website directly."

Context: 
---
{context_text}
---

Question: {query}
Answer:
"""
        try:
            chat_completion = self.llm_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.0
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            return f"Error during LLM generation: {e}"

    def get_stats(self) -> Dict[str, Any]:
        """Provides statistics about the loaded data."""
        total_chunks = len(self.chunks)
        unique_sources = len(set(chunk.get("source_url") for chunk in self.chunks))
        
        if total_chunks > 0:
            avg_length = sum(len(chunk["text"]) for chunk in self.chunks) / total_chunks
        else:
            avg_length = 0

        return {
            "total_chunks": total_chunks,
            "unique_sources": unique_sources,
            "average_chunk_length": round(avg_length, 2),
            "encoder_model": EMBEDDINGS_MODEL,
            "llm_model": "llama-3.3-70b-versatile"
        }

if __name__ == "__main__":
    # Example flow to ensure files are generated
    if not os.path.exists(CHUNKS_FILE) or not os.path.exists(EMBEDDINGS_FILE):
        print("Please run scraper.py and data_processor.py first to generate the necessary JSON files.")
    
    rag = RAGSystem(website_name="Amalitech")
    if rag.is_ready:
        test_queries = [
            "What programs are offered?",
            "How do I apply for admission?",
            "Where is the campus located?"
        ]
        
        for query in test_queries:
            print(f"\n--- QUERY: {query} ---")
            sources, chunks = rag.retrieve_context(query, top_k=3)
            
            if chunks:
                answer = rag.generate_answer(query, chunks)
                print(f"Answer: {answer}")
                print(f"Sources: {sources}")
            else:
                print("No relevant context found.")