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
    # Other fields can be added but ignored for now

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
# Initialize the FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="A Retrieval-Augmented Generation service for answering questions based on scraped website data.",
    version="1.0.0"
)

# Enable CORS for frontend applications (like the WordPress plugin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for simplicity in this example
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the RAG System globally
rag_system: Optional[RAGSystem] = None

# @app.on_event("startup")
# async def startup_event():
#     """Initializes the RAGSystem and loads embeddings on startup."""
#     global rag_system
#     print("Initializing RAG System...")
#     # NOTE: Set the name of the website here for the LLM prompt.
#     rag_system = RAGSystem(website_name="MPTI Ghana website")
#     if not rag_system.is_ready:
#         print("RAG System failed to initialize. Check if JSON files exist.")
#     else:
#         print("RAG System is ready to serve queries.")
def get_rag_system() -> RAGSystem:
    global rag_system
    if rag_system is None:
        print("Lazy-loading RAG system...")
        rag_system = RAGSystem(website_name="MPTI Ghana website")
        if not rag_system.is_ready:
            raise RuntimeError("RAG system failed to initialize")
    return rag_system
# --- Endpoints ---
@app.get("/", summary="Welcome Endpoint")
async def welcome():
    """Returns a welcome message and basic API information."""
    return {
        "message": "Welcome to the RAG Chatbot API.",
        "api_docs": "/docs",
        "health_check": "/health"
    }

# @app.get("/health", summary="System Health Check")
# async def health_check():
#     """Checks if the RAG system is initialized and ready."""
#     if not rag_system:
#         raise HTTPException(
#             status_code=503,
#             detail="RAG System is not initialized. Check logs for startup errors."
#         )
#     if not rag_system.is_ready:
#         raise HTTPException(
#             status_code=503,
#             detail="RAG System is initialized but not ready (embeddings or chunks missing)."
#         )
#     if not os.getenv("GROQ_API_KEY"):
#         raise HTTPException(
#             status_code=500,
#             detail="Groq API Key not configured in environment variables."
#         )

#     return {
#         "status": "Healthy",
#         "chunks_loaded": len(rag_system.chunks),
#         "llm_service": "Groq (llama-3.3-70b-versatile)",
#         "message": "System is running and ready to handle queries."
#     }
@app.get("/health")
async def health_check():
    try:
        rs = get_rag_system()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="Groq API key missing")

    return {
        "status": "Healthy",
        "chunks_loaded": len(rs.chunks),
    }

@app.get("/stats", summary="Data Statistics")
async def get_stats():
    """Returns statistics about the loaded processed data."""
    if not rag_system or not rag_system.is_ready:
         raise HTTPException(
            status_code=503,
            detail="RAG System is not ready. Cannot retrieve statistics."
        )
    return rag_system.get_stats()

# @app.post("/chat", response_model=ChatResponse, summary="Chat Endpoint")
# async def chat_with_rag_system(request: ChatRequest):
#     """
#     Processes a user message by retrieving relevant context and generating a reply
#     using the RAG pattern (Retrieval Augmented Generation).
#     """
#     if not rag_system or not rag_system.is_ready:
#          raise HTTPException(
#             status_code=503,
#             detail="RAG System is currently unavailable. Please try again later."
#         )
        
#     message = request.message
#     session_id = request.session_id or request.conversation_id or "default"

#     if not message:
#         raise HTTPException(status_code=400, detail="Message cannot be empty.")

#     try:
#         import time
#         start_time = time.time()
        
#         # 1. Retrieval
#         sources, chunks = rag_system.retrieve_context(message, 3)
        
#         if not chunks:
#             # If no context is found, try to generate a fallback answer
#             reply = "I don't have that information. Please check the website directly."
#         else:
#             # 2. Generation
#             reply = rag_system.generate_answer(message, chunks)

#         response_time = time.time() - start_time

#         # 3. Return response with plugin-expected format
#         return ChatResponse(
#             reply=reply,
#             session_id=session_id,
#             conversation_id=session_id,
#             intent="information_request",  # Dummy
#             confidence=0.8,  # Dummy
#             sentiment="neutral",  # Dummy
#             source="fastapi_backend",
#             response_time=response_time,
#             ctas=[],
#             external_links=[{"url": url, "text": url} for url in sources],
#             suggestions=[],
#             follow_up_questions=[],
#             nlp_analysis={"sentiment": "neutral", "entities": []}
#         )

#     except Exception as e:
#         print(f"An error occurred during message processing: {e}")
#         raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

@app.post("/chat", response_model=ChatResponse)
async def chat_with_rag_system(request: ChatRequest):
    try:
        rs = get_rag_system()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"RAG System is currently unavailable: {e}"
        )

    message = request.message
    session_id = request.session_id or request.conversation_id or "default"

    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # retrieval
    sources, chunks = rs.retrieve_context(message, 3)

    if not chunks:
        reply = "I don't have that information. Please check the website directly."
    else:
        reply = rs.generate_answer(message, chunks)

    return ChatResponse(
        reply=reply,
        session_id=session_id,
        conversation_id=session_id,
    )

if __name__ == "__main__":
    import uvicorn
    # Use $PORT environment variable if available (e.g., on Render)
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)