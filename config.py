# Configuration file for IT Helpdesk RAG Chatbot

import os
from dataclasses import dataclass

@dataclass
class Config:
    # OpenAI Configuration (for chat completions only)
    OPENAI_API_KEY: str = ""  # Replace with your key
    OPENAI_MODEL: str = "gpt-4o-mini"
    
    # Embedding Configuration (using local BGE model)
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    USE_LOCAL_EMBEDDINGS: bool = True  # Use local embeddings instead of OpenAI
    
    # ChromaDB Configuration
    CHROMA_DB_PATH: str = "./chroma_db"
    TICKETS_COLLECTION: str = "helpdesk_tickets"
    DOCS_COLLECTION: str = "pdf_documents"
    
    # Data Configuration
    EXCEL_FILE: str = "it_helpdesk_2000 Tickets.xlsx"
    PDF_CHUNK_SIZE: int = 750  # tokens
    PDF_CHUNK_OVERLAP: int = 100  # tokens
    
    # Search Configuration
    TOP_TICKETS: int = 10
    TOP_DOCS: int = 5
    RELEVANCE_THRESHOLD: float = 0.7
    
    # Fallback Configuration - MODIFY HERE FOR DIFFERENT FALLBACK BEHAVIOR
    FALLBACK_CRITERIA = {
        # Step 1: Tickets - Find up to 10 tickets with similarity > 60%
        "ticket_similarity_threshold": 0.6,  # 60% similarity minimum
        "max_tickets_to_check": 10,          # Check up to 10 tickets
        
        # Step 2: Documents - Find up to 3 docs with similarity > 50%
        "doc_similarity_threshold": 0.50,    # 50% similarity minimum  
        "max_docs_to_check": 3,              # Check up to 3 documents
        
        # Step 3: Web fallback
        "enable_web_fallback": True,         # Enable/disable web search
        "web_search_queries": 3,             # Number of web search results
        
        # Legacy (kept for relevance checking)
        "relevance_threshold": 0.6,          # For AI relevance assessment
    }
    
    # File paths
    DATA_DIR: str = "."
    LOGS_DIR: str = "./logs"

# Initialize config
config = Config()

# Validate API key
def validate_config():
    if not config.OPENAI_API_KEY:
        raise ValueError(
            "OpenAI API Key not found. Please set OPENAI_API_KEY environment variable "
            "or create a .env file with OPENAI_API_KEY=your_key_here"
        )
    return True
