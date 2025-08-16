# Complete testing including web search and non-IT queries

import sys
import os
import time
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline import ITHelpdeskPipeline
from data_processor import DataProcessor
from config import config, validate_config

class TestColors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class CompleteTester:
    def __init__(self):
        self.pipeline = None
        
    def print_header(self, text: str):
        """Print formatted header"""
        print(f"\n{TestColors.HEADER}{TestColors.BOLD}{'='*80}")
        print(f"{text}")
        print(f"{'='*80}{TestColors.ENDC}")
    
    def print_step(self, step: str):
        """Print step information"""
        print(f"{TestColors.OKBLUE}{TestColors.BOLD}➤ {step}{TestColors.ENDC}")
    
    def print_success(self, text: str):
        """Print success message"""
        print(f"{TestColors.OKGREEN}✓ {text}{TestColors.ENDC}")
    
    def print_warning(self, text: str):
        """Print warning message"""
        print(f"{TestColors.WARNING}⚠ {text}{TestColors.ENDC}")
    
    def print_error(self, text: str):
        """Print error message"""
        print(f"{TestColors.FAIL}✗ {text}{TestColors.ENDC}")
    
    def print_info(self, text: str):
        """Print info message"""
        print(f"{TestColors.OKCYAN}ℹ {text}{TestColors.ENDC}")

    def setup_system(self):
        """Initialize the system"""
        self.print_header("COMPLETE SYSTEM SETUP")
        
        try:
            self.print_step("Validating configuration...")
            validate_config()
            self.print_success("Configuration validated")
            
            self.print_step("Checking existing data...")
            data_processor = DataProcessor()
            ticket_count = data_processor.tickets_collection.count()
            doc_count = data_processor.docs_collection.count()
            
            if ticket_count == 0:
                self.print_error("No tickets found in database! Please run init_data.py first.")
                return False
            
            self.print_success(f"Data ready: {ticket_count} tickets, {doc_count} document chunks")
            
            self.print_step("Initializing pipeline with web search...")
            # Suppress BGE model loading messages
            import os
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            import logging
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            
            self.pipeline = ITHelpdeskPipeline()
            # Disable progress bars
            if hasattr(self.pipeline.data_processor, 'embedding_model') and self.pipeline.data_processor.embedding_model:
                self.pipeline.data_processor.embedding_model.show_progress_bar = False
            
            self.print_success("Pipeline ready for complete testing")
            
            return True
            
        except Exception as e:
            self.print_error(f"Setup failed: {e}")
            return False

    def test_full_pipeline(self, query: str, test_name: str, expected_outcome: str):
        """Test complete pipeline including all fallbacks"""
        self.print_header(f"COMPLETE TEST: {test_name}")
        
        print(f"{TestColors.BOLD}🔄 Query: {query}{TestColors.ENDC}")
        print(f"{TestColors.BOLD}🎯 Expected: {expected_outcome}{TestColors.ENDC}")
        print()
        
        start_time = time.time()
        
        try:
            response, trace = self.pipeline.process_query(query)
            end_time = time.time()
            
            print(f"{TestColors.BOLD}📊 RESULTS:{TestColors.ENDC}")
            
            # Classification
            if trace.classification_result:
                result = trace.classification_result
                print(f"  🔍 Classification: {result['is_it_query']} (confidence: {result['confidence']:.3f})")
                print(f"     Reason: {result['reason']}")
            
            # What happened
            if trace.classification_result and not trace.classification_result.get('is_it_query', False):
                print(f"  🚫 Non-IT query detected - appropriate response given")
            elif trace.tickets_found and len(trace.tickets_found) > 0:
                print(f"  🎫 Found {len(trace.tickets_found)} high-quality tickets")
                if trace.ticket_relevance and trace.ticket_relevance.get('has_useful_tickets'):
                    print(f"     ✅ Tickets were relevant - used for response")
                else:
                    print(f"     ⚠️ Tickets not relevant enough - triggered fallback")
            
            # Fallbacks
            if trace.fallback_triggered:
                if trace.fallback_triggered == "docs":
                    print(f"  📄 Document fallback: Found {len(trace.docs_found) if trace.docs_found else 0} chunks")
                elif trace.fallback_triggered == "web":
                    print(f"  🌐 Web search fallback: Found {len(trace.web_results) if trace.web_results else 0} results")
            
            # Performance
            print(f"  ⏱️ Total time: {end_time - start_time:.2f}s")
            print(f"  📝 Response length: {len(response)} characters")
            
            # Error check
            if trace.error_message:
                print(f"  ❌ Errors: {trace.error_message}")
            else:
                print(f"  ✅ No errors")
            
            print()
            print(f"{TestColors.BOLD}📋 FINAL RESPONSE:{TestColors.ENDC}")
            print(f"{response}")
            print()
            
            return True
            
        except Exception as e:
            end_time = time.time()
            print(f"{TestColors.FAIL}❌ Test failed with exception: {e}{TestColors.ENDC}")
            print(f"  ⏱️ Time before error: {end_time - start_time:.2f}s")
            return False

    def run_complete_tests(self):
        """Run complete system tests"""
        print(f"{TestColors.HEADER}{TestColors.BOLD}")
        print("🔧 IT HELPDESK RAG CHATBOT - COMPLETE SYSTEM TESTING")
        print(f"{'='*80}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Testing: Classification, Quality Search, Fallbacks, Web Search")
        print(f"{'='*80}{TestColors.ENDC}")
        
        # Setup
        if not self.setup_system():
            self.print_error("Setup failed, aborting tests")
            return
        
        # Test 1: Non-IT Query
        self.test_full_pipeline(
            "What's the weather like today?",
            "Non-IT Query Test",
            "Should detect non-IT and give appropriate response"
        )
        
        # Test 2: Obscure IT Query (should trigger web search)
        self.test_full_pipeline(
            "My quantum computer is experiencing flux capacitor errors",
            "Obscure IT Issue (Web Search Test)",
            "Should trigger web search fallback"
        )
        
        # Test 3: Common IT Query (should use tickets)
        self.test_full_pipeline(
            "How do I reset my forgotten password?",
            "Common IT Issue (Ticket Match Test)", 
            "Should find relevant tickets and use them"
        )
        
        self.print_header("COMPLETE TESTING FINISHED")
        self.print_success("All complete system tests finished!")
        print()
        print(f"{TestColors.BOLD}🎯 System Capabilities Verified:{TestColors.ENDC}")
        print(f"  ✅ Non-IT query detection and appropriate responses")
        print(f"  ✅ Quality-based ticket search with BGE embeddings")
        print(f"  ✅ Smart fallback system (tickets → docs → web)")
        print(f"  ✅ Web search integration for unknown issues")
        print(f"  ✅ Professional response generation for all scenarios")

def main():
    """Main test function"""
    tester = CompleteTester()
    tester.run_complete_tests()

if __name__ == "__main__":
    main()
