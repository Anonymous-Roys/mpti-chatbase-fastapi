import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from rag_system import RAGSystem

# Load environment variables
load_dotenv()

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    """Request model for the chat endpoint."""
    message: str = Field(..., description="The user's message to the chatbot.")
    session_id: Optional[str] = Field(None, description="Session ID for conversation context.")
    conversation_id: Optional[str] = Field(None, description="Conversation ID.")

class Link(BaseModel):
    url: str
    text: str

class ChatResponse(BaseModel):
    """Response model for the chat endpoint."""
    reply: str = Field(..., description="The chatbot's reply.")
    session_id: Optional[str] = Field(None, description="Session ID.")
    conversation_id: Optional[str] = Field(None, description="Conversation ID.")
    intent: Optional[str] = Field(None, description="Detected intent.")
    confidence: Optional[float] = Field(0.0, description="Confidence score.")
    sentiment: Optional[str] = Field(None, description="Sentiment analysis.")
    source: str = Field("fastapi_backend", description="Source of the response.")
    response_time: Optional[float] = Field(0.0, description="Response time in seconds.")
    ctas: List[str] = Field([], description="Call to actions.")
    external_links: List[Link] = Field([], description="External links.")
    suggestions: List[str] = Field([], description="Suggestions.")
    follow_up_questions: List[str] = Field([], description="Follow-up questions.")
    nlp_analysis: Dict[str, Any] = Field({}, description="NLP analysis results.")

# --- FastAPI Initialization ---
app = FastAPI(
    title="RAG Chatbot API",
    description="A Retrieval-Augmented Generation service for answering questions based on scraped website data.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG System instance (initialized lazily)
rag_system: Optional[RAGSystem] = None

@app.on_event("startup")
async def startup_event():
    """Minimal startup - RAG system loads on first request."""
    global rag_system
    print("Server starting up...")
    print("RAG System will be initialized on first request to save memory.")
    # Don't initialize RAG system here - save memory!
    rag_system = None

def get_rag_system() -> RAGSystem:
    global rag_system
    if rag_system is None:
        print("Starting RAG system initialization...")
        # Add timeout to initialization
        rag_system = RAGSystem(website_name="MPTI Ghana website")
        print(f"RAG System loaded. Chunks: {len(rag_system.chunks)}")
    return rag_system


# --- Endpoints ---
@app.get("/", summary="Welcome Endpoint")
async def welcome():
    """Returns a welcome message and basic API information."""
    return {
        "message": "Welcome to the RAG Chatbot API.",
        "api_docs": "/docs",
        "health_check": "/health",
        "note": "RAG system loads on first chat request to optimize memory usage."
    }


@app.get("/health")
async def health_check():
    # Check if Groq API key is configured
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="Groq API Key not configured.")
    # Lightweight file checks
    chunks_exist = os.path.exists("chunks.json")
    embeddings_exist = os.path.exists("embeddings.json")
    return {
        "status": "Ready" if (chunks_exist and embeddings_exist) else "Waiting for data",
        "rag_system_loaded": rag_system is not None,  # Just check, don't load
        "chunks_file_exists": chunks_exist,
        "embeddings_file_exists": embeddings_exist,
        "llm_service": "Groq"
    }

@app.get("/stats", summary="Data Statistics")
async def get_stats():
    """
    Returns statistics about the loaded data.
    This will trigger lazy loading if RAG system isn't initialized yet.
    """
    try:
        rs = get_rag_system()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot retrieve statistics - RAG System unavailable: {e}"
        )
    
    return rs.get_stats()

@app.post("/chat", response_model=ChatResponse, summary="Chat Endpoint")
async def chat_with_rag_system(request: ChatRequest):
    """
    Processes a user message using RAG pattern.
    Lazy-loads the RAG system on first request.
    """
    # Lazy load RAG system
    try:
        rs = get_rag_system()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"RAG System is currently unavailable: {e}"
        )
    
    message = request.message
    session_id = request.session_id or request.conversation_id or "default"

    if not message or not message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        import time
        start_time = time.time()
        
        # 1. Retrieval
        sources, chunks = rs.retrieve_context(message, top_k=3)
        
        if not chunks:
            reply = "I don't have that information in my knowledge base. Please check the website directly or rephrase your question."
        else:
            # 2. Generation
            reply = rs.generate_answer(message, chunks)

        response_time = time.time() - start_time

        # 3. Return response
        return ChatResponse(
            reply=reply,
            session_id=session_id,
            conversation_id=session_id,
            intent="information_request",
            confidence=0.85 if chunks else 0.3,
            sentiment="neutral",
            source="fastapi_backend",
            response_time=response_time,
            ctas=[],
            external_links=[Link(url=url, text=url) for url in sources[:3]],  # Limit to 3 sources
            suggestions=[],
            follow_up_questions=[],
            nlp_analysis={"sentiment": "neutral", "entities": [], "sources_count": len(sources)}
        )

    except Exception as e:
        print(f"Error during message processing: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal Server Error: {str(e)}"
        )

@app.get("/memory", summary="Memory Usage Stats")
async def memory_stats():
    """
    Returns current memory usage statistics.
    Useful for monitoring and debugging memory issues.
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "memory_mb": round(memory_info.rss / 1024 / 1024, 2),
            "memory_percent": round(process.memory_percent(), 2),
            "rag_system_loaded": rag_system is not None,
            "chunks_count": len(rag_system.chunks) if rag_system else 0
        }
    except ImportError:
        return {
            "error": "psutil not installed. Run: pip install psutil",
            "rag_system_loaded": rag_system is not None
        }

if __name__ == "__main__":
    import uvicorn
    # Use $PORT environment variable (required by Render)
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)