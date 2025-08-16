# Data initialization script for IT Helpdesk RAG Chatbot

import os
import sys
from data_processor import DataProcessor
from config import validate_config

def main():
    """Initialize the data for the chatbot"""
    print("🔧 IT Helpdesk RAG Chatbot - Data Initialization")
    print("="*60)
    
    try:
        # Validate configuration
        print("1. Validating configuration...")
        validate_config()
        print("   ✓ Configuration valid")
        print(f"   ℹ Using BGE-small-en-v1.5 for embeddings (local)")
        
        # Check for required files
        print("2. Checking for required files...")
        excel_file = "it_helpdesk_2000 Tickets.xlsx"
        if not os.path.exists(excel_file):
            print(f"   ✗ Missing: {excel_file}")
            print("   Please ensure the Excel file is in the current directory")
            return False
        else:
            print(f"   ✓ Found: {excel_file}")
        
        # Check for PDF files
        pdf_files = [f for f in os.listdir(".") if f.endswith('.pdf')]
        if not pdf_files:
            print("   ⚠ No PDF files found - documentation search will be limited")
        else:
            print(f"   ✓ Found {len(pdf_files)} PDF files")
        
        # Initialize data processor with suppressed logging
        print("3. Initializing data processor...")
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        import logging
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        
        processor = DataProcessor()
        print("   ✓ Data processor ready")
        
        # Ingest tickets
        print("4. Processing tickets with BGE embeddings...")
        print("   ℹ This will process 2000 tickets efficiently in batches")
        processor.ingest_tickets(force_reindex=True)
        ticket_count = processor.tickets_collection.count()
        print(f"   ✓ Indexed {ticket_count} tickets")
        
        # Ingest PDFs
        if pdf_files:
            print("5. Processing PDF documents...")
            processor.ingest_pdfs(force_reindex=True)
            doc_count = processor.docs_collection.count()
            print(f"   ✓ Indexed {doc_count} document chunks")
        else:
            print("5. Skipping PDF processing (no PDFs found)")
        
        print("\n🎉 Data initialization completed successfully!")
        print("\nNext steps:")
        print("  • Run: streamlit run app.py")
        print("  • Or test: python test_inference.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Initialization failed: {e}")
        print("\nTroubleshooting:")
        print("  • Check that OPENAI_API_KEY is set in config.py")
        print("  • Ensure all required files are present")
        print("  • Run: pip install -r requirements.txt")
        return False

if __name__ == "__main__":
    main()
