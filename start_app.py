# Simple startup script for the IT Helpdesk Chat App

import subprocess
import sys
import os

def check_data():
    """Check if data is initialized"""
    try:
        from data_processor import DataProcessor
        processor = DataProcessor()
        ticket_count = processor.tickets_collection.count()
        
        if ticket_count == 0:
            print("⚠ No data found! Please run: python init_data.py")
            return False
        else:
            print(f"✓ Data ready: {ticket_count} tickets indexed")
            return True
    except Exception as e:
        print(f"❌ Error checking data: {e}")
        return False

def main():
    """Main startup function"""
    print("🔧 IT Helpdesk RAG Chatbot - Starting...")
    print("="*50)
    
    # Check if data is ready
    if not check_data():
        print("\nPlease initialize data first:")
        print("python init_data.py")
        return
    
    # Start Streamlit app
    print("\n🚀 Starting Streamlit chat interface...")
    print("Opening in your default browser...")
    print("Use Ctrl+C to stop the server")
    print("="*50)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")

if __name__ == "__main__":
    main()
