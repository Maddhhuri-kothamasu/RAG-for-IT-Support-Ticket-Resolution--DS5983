# Data processing module for ingesting Excel and PDF files

import pandas as pd
import PyPDF2
import chromadb
import openai
from typing import List, Dict, Any
import os
import json
import logging
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from config import config, validate_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TicketData:
    id: str
    subject: str
    description: str
    resolution_strategy: str
    inferred_topic: str
    category: str
    priority: str
    created_at: str
    requester_email: str
    combined_text: str

@dataclass
class DocumentChunk:
    text: str
    source: str
    page: int
    chunk_id: str

class DataProcessor:
    def __init__(self):
        validate_config()
        
        # Initialize OpenAI for chat completions only
        self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        
        # Initialize embedding model (local BGE or OpenAI)
        if config.USE_LOCAL_EMBEDDINGS:
            logger.info(f"Loading local embedding model: {config.EMBEDDING_MODEL}")
            # Suppress progress bars and warnings
            import os
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            import logging
            logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
            
            self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
            # Disable progress bars for encoding
            self.embedding_model.show_progress_bar = False
            logger.info("Local embedding model loaded successfully")
        else:
            logger.info("Using OpenAI embeddings")
            self.embedding_model = None
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        
        # Get or create collections
        try:
            self.tickets_collection = self.chroma_client.get_collection(name=config.TICKETS_COLLECTION)
            logger.info("Loaded existing tickets collection")
        except:
            self.tickets_collection = self.chroma_client.create_collection(name=config.TICKETS_COLLECTION)
            logger.info("Created new tickets collection")
            
        try:
            self.docs_collection = self.chroma_client.get_collection(name=config.DOCS_COLLECTION)
            logger.info("Loaded existing docs collection")
        except:
            self.docs_collection = self.chroma_client.create_collection(name=config.DOCS_COLLECTION)
            logger.info("Created new docs collection")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using local BGE model or OpenAI"""
        try:
            if config.USE_LOCAL_EMBEDDINGS:
                # Use local BGE model
                embedding = self.embedding_model.encode([text], normalize_embeddings=True)[0]
                return embedding.tolist()
            else:
                # Use OpenAI embeddings
                response = self.openai_client.embeddings.create(
                    model=config.EMBEDDING_MODEL,
                    input=text
                )
                return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def load_tickets_from_excel(self) -> List[TicketData]:
        """Load and process tickets from Excel file"""
        try:
            df = pd.read_excel(config.EXCEL_FILE)
            tickets = []
            
            for _, row in df.iterrows():
                # Combine key columns for embedding
                combined_text = f"""
                Subject: {row.get('subject', '')}
                Description: {row.get('description', '')}
                Resolution: {row.get('resolution_strategy', '')}
                Topic: {row.get('inferred_topic', '')}
                Category: {row.get('category', '')}
                """.strip()
                
                ticket = TicketData(
                    id=str(row.get('id', '')),
                    subject=str(row.get('subject', '')),
                    description=str(row.get('description', '')),
                    resolution_strategy=str(row.get('resolution_strategy', '')),
                    inferred_topic=str(row.get('inferred_topic', '')),
                    category=str(row.get('category', '')),
                    priority=str(row.get('priority', '')),
                    created_at=str(row.get('createdat', '')),
                    requester_email=str(row.get('requesteremail', '')),
                    combined_text=combined_text
                )
                tickets.append(ticket)
                
            logger.info(f"Loaded {len(tickets)} tickets from Excel")
            return tickets
            
        except Exception as e:
            logger.error(f"Error loading tickets: {e}")
            raise

    def process_pdf_file(self, pdf_path: str) -> List[DocumentChunk]:
        """Process a single PDF file into chunks"""
        chunks = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    
                    # Clean and normalize text
                    text = text.replace('\n', ' ').replace('\r', ' ')
                    text = ' '.join(text.split())  # Remove extra whitespace
                    
                    if len(text.strip()) < 100:  # Skip pages with too little content
                        continue
                    
                    # Simple chunking by character count (approximate tokens)
                    chunk_size = config.PDF_CHUNK_SIZE * 4  # ~4 chars per token
                    overlap = config.PDF_CHUNK_OVERLAP * 4
                    
                    for i in range(0, len(text), chunk_size - overlap):
                        chunk_text = text[i:i + chunk_size]
                        if len(chunk_text.strip()) < 50:  # Skip very small chunks
                            continue
                            
                        chunk = DocumentChunk(
                            text=chunk_text.strip(),
                            source=os.path.basename(pdf_path),
                            page=page_num + 1,
                            chunk_id=f"{os.path.basename(pdf_path)}_{page_num}_{i}"
                        )
                        chunks.append(chunk)
                        
            logger.info(f"Processed {pdf_path}: {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return []

    def ingest_tickets(self, force_reindex: bool = False):
        """Ingest tickets into ChromaDB"""
        
        # Check if collection already has data
        if not force_reindex and self.tickets_collection.count() > 0:
            logger.info("Tickets already indexed. Use force_reindex=True to re-index")
            return
            
        logger.info("Starting ticket ingestion...")
        tickets = self.load_tickets_from_excel()
        
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        logger.info(f"Generating embeddings for {len(tickets)} tickets...")
        
        if config.USE_LOCAL_EMBEDDINGS:
            # Batch processing for local embeddings (more efficient)
            texts = [ticket.combined_text for ticket in tickets]
            embeddings = self.embedding_model.encode(texts, 
                                                   normalize_embeddings=True, 
                                                   show_progress_bar=False,
                                                   batch_size=32)  # Process in batches
            embeddings = [emb.tolist() for emb in embeddings]
        else:
            # Individual processing for OpenAI embeddings
            embeddings = []
            for i, ticket in enumerate(tickets):
                if i % 50 == 0:  # Progress update every 50 tickets
                    logger.info(f"Processing ticket {i+1}/{len(tickets)}")
                embedding = self.generate_embedding(ticket.combined_text)
                embeddings.append(embedding)
        
        # Prepare metadata and IDs
        for ticket in tickets:
            documents.append(ticket.combined_text)
            metadatas.append({
                "id": ticket.id,
                "subject": ticket.subject,
                "inferred_topic": ticket.inferred_topic,
                "category": ticket.category,
                "priority": ticket.priority,
                "resolution_strategy": ticket.resolution_strategy
            })
            ids.append(f"ticket_{ticket.id}")
            
        # Add to collection
        self.tickets_collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        
        logger.info(f"Ingested {len(tickets)} tickets into ChromaDB")

    def ingest_pdfs(self, force_reindex: bool = False):
        """Ingest PDF documents into ChromaDB"""
        
        # Check if collection already has data
        if not force_reindex and self.docs_collection.count() > 0:
            logger.info("PDFs already indexed. Use force_reindex=True to re-index")
            return
            
        logger.info("Starting PDF ingestion...")
        
        # Find all PDF files
        pdf_files = [f for f in os.listdir(config.DATA_DIR) if f.endswith('.pdf')]
        
        if not pdf_files:
            logger.warning("No PDF files found in data directory")
            return
            
        all_chunks = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(config.DATA_DIR, pdf_file)
            chunks = self.process_pdf_file(pdf_path)
            all_chunks.extend(chunks)
            
        if not all_chunks:
            logger.warning("No chunks extracted from PDFs")
            return
            
        logger.info(f"Generating embeddings for {len(all_chunks)} document chunks...")
        
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        if config.USE_LOCAL_EMBEDDINGS:
            # Batch processing for local embeddings
            texts = [chunk.text for chunk in all_chunks]
            embeddings = self.embedding_model.encode(texts, 
                                                   normalize_embeddings=True, 
                                                   show_progress_bar=False,
                                                   batch_size=16)  # Smaller batch for docs
            embeddings = [emb.tolist() for emb in embeddings]
        else:
            # Individual processing for OpenAI embeddings
            embeddings = []
            for i, chunk in enumerate(all_chunks):
                if i % 20 == 0:  # Progress update every 20 chunks
                    logger.info(f"Processing chunk {i+1}/{len(all_chunks)}")
                embedding = self.generate_embedding(chunk.text)
                embeddings.append(embedding)
        
        # Prepare metadata and IDs
        for chunk in all_chunks:
            documents.append(chunk.text)
            metadatas.append({
                "source": chunk.source,
                "page": chunk.page,
                "chunk_id": chunk.chunk_id
            })
            ids.append(chunk.chunk_id)
            
        # Add to collection
        self.docs_collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        
        logger.info(f"Ingested {len(all_chunks)} document chunks into ChromaDB")

    def search_tickets(self, query: str, n_results: int = None) -> Dict[str, Any]:
        """Search for similar tickets"""
        if n_results is None:
            n_results = config.TOP_TICKETS
            
        try:
            query_embedding = self.generate_embedding(query)
            results = self.tickets_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            return {
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "distances": results["distances"][0] if results["distances"] else []
            }
            
        except Exception as e:
            logger.error(f"Error searching tickets: {e}")
            return {"documents": [], "metadatas": [], "distances": []}

    def search_documents(self, query: str, n_results: int = None) -> Dict[str, Any]:
        """Search for similar document chunks"""
        if n_results is None:
            n_results = config.TOP_DOCS
            
        try:
            query_embedding = self.generate_embedding(query)
            results = self.docs_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            return {
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "distances": results["distances"][0] if results["distances"] else []
            }
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return {"documents": [], "metadatas": [], "distances": []}

def main():
    """Main function for testing data processing"""
    processor = DataProcessor()
    
    print("Ingesting tickets...")
    processor.ingest_tickets()
    
    print("Ingesting PDFs...")
    processor.ingest_pdfs()
    
    print("Testing search...")
    results = processor.search_tickets("printer not working")
    print(f"Found {len(results['documents'])} tickets")
    
    results = processor.search_documents("VPN setup")
    print(f"Found {len(results['documents'])} document chunks")
    
    print("Data processing completed!")

if __name__ == "__main__":
    main()
