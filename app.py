# Clean Chatbot Interface for IT Helpdesk RAG Chatbot

import streamlit as st
import time
from datetime import datetime
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline import ITHelpdeskPipeline
from data_processor import DataProcessor
from config import config, validate_config

# Page configuration
st.set_page_config(
    page_title="IT Helpdesk Assistant",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'pipeline' not in st.session_state:
        try:
            validate_config()
            
            # Suppress BGE model loading messages
            import os
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            import logging
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            
            st.session_state.pipeline = ITHelpdeskPipeline()
            st.session_state.data_processor = DataProcessor()
            
            # Disable progress bars
            if hasattr(st.session_state.pipeline.data_processor, 'embedding_model') and st.session_state.pipeline.data_processor.embedding_model:
                st.session_state.pipeline.data_processor.embedding_model.show_progress_bar = False
                
        except Exception as e:
            st.error(f"Failed to initialize pipeline: {e}")
            st.stop()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def display_trace_step(step_name: str, content: str, step_type: str = "info"):
    """Display a trace step with appropriate styling"""
    if step_type == "success":
        st.success(f"**{step_name}**")
        st.write(content)
    elif step_type == "warning":
        st.warning(f"**{step_name}**")
        st.write(content)
    elif step_type == "error":
        st.error(f"**{step_name}**")
        st.write(content)
    else:
        st.info(f"**{step_name}**")
        st.write(content)

def process_query_with_trace(user_query: str):
    """Process query and display live trace"""
    
    # Create trace container
    trace_container = st.container()
    
    with trace_container:
        st.subheader("Processing Trace")
        
        # Initialize trace object
        trace = type('', (), {})()
        trace.original_query = user_query
        
        start_time = time.time()
        
        try:
            # Step 1: Classification
            step1_placeholder = st.empty()
            with step1_placeholder.container():
                st.info("**Step 1: Classifying query...**")
                st.write("Determining if this is an IT-related query...")
            
            is_it = st.session_state.pipeline.step1_classify_query(user_query, trace)
            
            step1_placeholder.empty()
            if hasattr(trace, 'classification_result') and trace.classification_result:
                result = trace.classification_result
                if result['is_it_query']:
                    display_trace_step(
                        "Step 1: Query Classification ✓", 
                        f"**Result:** {result['is_it_query']} (confidence: {result['confidence']:.3f})\n\n**Reason:** {result['reason']}", 
                        "success"
                    )
                else:
                    display_trace_step(
                        "Step 1: Query Classification ✗", 
                        f"**Result:** {result['is_it_query']} (confidence: {result['confidence']:.3f})\n\n**Reason:** {result['reason']}\n\n**Action:** Stopping here - not an IT query", 
                        "warning"
                    )
                    return st.session_state.pipeline.prompts.get_non_it_response(), trace, time.time() - start_time
            
            # Step 2: Query Expansion
            step2_placeholder = st.empty()
            with step2_placeholder.container():
                st.info("**Step 2: Expanding query...**")
                st.write("Adding synonyms and related terms...")
            
            expanded = st.session_state.pipeline.step2_expand_query(user_query, trace)
            
            step2_placeholder.empty()
            display_trace_step(
                "Step 2: Query Expansion ✓", 
                f"**Original:** {user_query}\n\n**Expanded:** {expanded[:200]}...", 
                "info"
            )
            
            # Step 3: Search Tickets
            step3_placeholder = st.empty()
            with step3_placeholder.container():
                st.info("**Step 3: Searching tickets...**")
                st.write("Finding similar historical tickets using BGE embeddings...")
            
            tickets = st.session_state.pipeline.step3_search_tickets(expanded, trace)
            
            step3_placeholder.empty()
            threshold = config.FALLBACK_CRITERIA["ticket_similarity_threshold"]
            
            if tickets:
                ticket_details = f"**Quality threshold:** >={threshold:.0%} similarity\n\n**Found {len(tickets)} high-quality tickets:**\n\n"
                for i, ticket in enumerate(tickets[:3]):
                    ticket_details += f"{i+1}. {ticket['subject']} ({ticket['similarity_score']:.1%})\n"
                
                display_trace_step("Step 3: Ticket Search ✓", ticket_details, "success")
            else:
                display_trace_step(
                    "Step 3: Ticket Search ⚠", 
                    f"**Quality threshold:** >={threshold:.0%} similarity\n\nNo high-quality tickets found", 
                    "warning"
                )
            
            # Step 4: Relevance Check
            if tickets:
                step4_placeholder = st.empty()
                with step4_placeholder.container():
                    st.info("**Step 4: Checking relevance...**")
                    st.write("AI is evaluating ticket relevance...")
                
                relevant, filtered = st.session_state.pipeline.step4_check_relevance(user_query, tickets, trace)
                
                step4_placeholder.empty()
                if hasattr(trace, 'ticket_relevance') and trace.ticket_relevance:
                    rel = trace.ticket_relevance
                    
                    relevance_details = f"**Overall Relevance:** {rel.get('overall_relevance', 0):.3f}\n\n"
                    relevance_details += f"**Has Useful Tickets:** {rel.get('has_useful_tickets', False)}\n\n"
                    relevance_details += f"**Recommendation:** {rel.get('recommendation', 'N/A')}"
                    
                    if relevant:
                        relevance_details += "\n\n**Action:** Using tickets for final response!"
                        display_trace_step("Step 4: Relevance Check ✓", relevance_details, "success")
                        
                        # Generate response from tickets
                        response_placeholder = st.empty()
                        with response_placeholder.container():
                            st.info("**Step 7: Generating response from tickets...**")
                        
                        response = st.session_state.pipeline.step7_generate_response(user_query, filtered, [], [], trace)
                        
                        response_placeholder.empty()
                        end_time = time.time()
                        
                        display_trace_step(
                            "Step 7: Response Generation ✓", 
                            f"**Response generated:** {len(response)} characters\n\n**Total time:** {end_time - start_time:.2f}s", 
                            "success"
                        )
                        
                        return response, trace, end_time - start_time
                    else:
                        relevance_details += "\n\n**Action:** Tickets not relevant enough, trying documents..."
                        display_trace_step("Step 4: Relevance Check ⚠", relevance_details, "warning")
            
            # Step 5: Document Search (Fallback)
            step5_placeholder = st.empty()
            with step5_placeholder.container():
                st.warning("**Step 5: Searching documents (fallback)...**")
                st.write("Looking for relevant documentation...")
            
            docs = st.session_state.pipeline.step5_search_documents(expanded, trace)
            
            step5_placeholder.empty()
            doc_threshold = config.FALLBACK_CRITERIA["doc_similarity_threshold"]
            
            if docs:
                doc_details = f"**Quality threshold:** >={doc_threshold:.0%} similarity\n\n**Found {len(docs)} high-quality document chunks:**\n\n"
                for i, doc in enumerate(docs):
                    doc_details += f"{i+1}. {doc['source']} (page {doc['page']}) - {doc['similarity_score']:.1%}\n"
                
                display_trace_step("Step 5: Document Search ✓", doc_details, "success")
                
                # Generate response from docs
                response_placeholder = st.empty()
                with response_placeholder.container():
                    st.info("**Step 7: Generating response from documents...**")
                
                response = st.session_state.pipeline.step7_generate_response(user_query, [], docs, [], trace)
                
                response_placeholder.empty()
                end_time = time.time()
                
                display_trace_step(
                    "Step 7: Response Generation ✓", 
                    f"**Response generated:** {len(response)} characters\n\n**Total time:** {end_time - start_time:.2f}s", 
                    "success"
                )
                
                return response, trace, end_time - start_time
            else:
                display_trace_step(
                    "Step 5: Document Search ⚠", 
                    f"**Quality threshold:** >={doc_threshold:.0%} similarity\n\nNo high-quality document chunks found", 
                    "warning"
                )
            
            # Step 6: Web Search (Final Fallback)
            if config.FALLBACK_CRITERIA["enable_web_fallback"]:
                step6_placeholder = st.empty()
                with step6_placeholder.container():
                    st.error("**Step 6: Web search (final fallback)...**")
                    st.write("Searching the web for solutions...")
                
                web_results = st.session_state.pipeline.step6_web_search(user_query, trace)
                
                step6_placeholder.empty()
                
                if web_results:
                    web_details = f"**Found {len(web_results)} web results:**\n\n"
                    for i, result in enumerate(web_results):
                        web_details += f"{i+1}. {result['title']}\n"
                    
                    display_trace_step("Step 6: Web Search ✓", web_details, "info")
                else:
                    display_trace_step("Step 6: Web Search ⚠", "No web results found", "warning")
                
                # Generate response from web
                response_placeholder = st.empty()
                with response_placeholder.container():
                    st.info("**Step 7: Generating response from web results...**")
                
                response = st.session_state.pipeline.step7_generate_response(user_query, [], [], web_results, trace)
                
                response_placeholder.empty()
                end_time = time.time()
                
                display_trace_step(
                    "Step 7: Response Generation ✓", 
                    f"**Response generated:** {len(response)} characters\n\n**Total time:** {end_time - start_time:.2f}s", 
                    "success"
                )
                
                return response, trace, end_time - start_time
            else:
                end_time = time.time()
                display_trace_step(
                    "Step 6: Web Search ✗", 
                    f"Web fallback is disabled\n\n**Total time:** {end_time - start_time:.2f}s", 
                    "error"
                )
                
                return "I apologize, but I couldn't find relevant information to help with your query. Please contact IT support directly.", trace, end_time - start_time
        
        except Exception as e:
            end_time = time.time()
            display_trace_step(
                "Error", 
                f"An error occurred during processing: {str(e)}\n\n**Time before error:** {end_time - start_time:.2f}s", 
                "error"
            )
            return "I apologize, but I encountered an error while processing your request. Please contact IT support directly.", trace, end_time - start_time

def process_query_simple(user_query: str):
    """Process query and return response with trace data as text"""
    
    start_time = time.time()
    trace_steps = []
    
    try:
        # Initialize trace object
        trace = type('', (), {})()
        trace.original_query = user_query
        
        # Step 1: Classification
        trace_steps.append("**Step 1: Query Classification**")
        is_it = st.session_state.pipeline.step1_classify_query(user_query, trace)
        
        if hasattr(trace, 'classification_result') and trace.classification_result:
            result = trace.classification_result
            trace_steps.append(f"- Result: {result['is_it_query']} (confidence: {result['confidence']:.3f})")
            trace_steps.append(f"- Reason: {result['reason']}")
            
            if not result['is_it_query']:
                trace_steps.append("- Action: Not an IT query, providing appropriate response")
                end_time = time.time()
                trace_data = "\n".join(trace_steps)
                return st.session_state.pipeline.prompts.get_non_it_response(), trace_data, end_time - start_time
        
        # Step 2: Query Expansion
        trace_steps.append("\n**Step 2: Query Expansion**")
        expanded = st.session_state.pipeline.step2_expand_query(user_query, trace)
        trace_steps.append(f"- Original: {user_query}")
        trace_steps.append(f"- Expanded: {expanded[:150]}...")
        
        # Step 3: Search Tickets
        trace_steps.append("\n**Step 3: Ticket Search**")
        tickets = st.session_state.pipeline.step3_search_tickets(expanded, trace)
        threshold = config.FALLBACK_CRITERIA["ticket_similarity_threshold"]
        trace_steps.append(f"- Quality threshold: >={threshold:.0%} similarity")
        trace_steps.append(f"- Found {len(tickets)} high-quality tickets")
        
        if tickets:
            for i, ticket in enumerate(tickets[:3]):
                trace_steps.append(f"  {i+1}. {ticket['subject']} ({ticket['similarity_score']:.1%})")
        
        # Step 4: Relevance Check
        if tickets:
            trace_steps.append("\n**Step 4: Relevance Check**")
            relevant, filtered = st.session_state.pipeline.step4_check_relevance(user_query, tickets, trace)
            
            if hasattr(trace, 'ticket_relevance') and trace.ticket_relevance:
                rel = trace.ticket_relevance
                trace_steps.append(f"- Overall Relevance: {rel.get('overall_relevance', 0):.3f}")
                trace_steps.append(f"- Has Useful Tickets: {rel.get('has_useful_tickets', False)}")
                trace_steps.append(f"- Recommendation: {rel.get('recommendation', 'N/A')}")
                
                if relevant:
                    trace_steps.append("- Action: Using tickets for response")
                    response = st.session_state.pipeline.step7_generate_response(user_query, filtered, [], [], trace)
                    end_time = time.time()
                    trace_steps.append(f"\n**Response Generation**")
                    trace_steps.append(f"- Generated {len(response)} characters in {end_time - start_time:.2f}s")
                    trace_data = "\n".join(trace_steps)
                    return response, trace_data, end_time - start_time
                else:
                    trace_steps.append("- Action: Tickets not relevant, trying documents")
        else:
            trace_steps.append("\n**Step 4: Relevance Check**")
            trace_steps.append("- No tickets to check, proceeding to documents")
        
        # Step 5: Document Search
        trace_steps.append("\n**Step 5: Document Search (Fallback)**")
        docs = st.session_state.pipeline.step5_search_documents(expanded, trace)
        doc_threshold = config.FALLBACK_CRITERIA["doc_similarity_threshold"]
        trace_steps.append(f"- Quality threshold: >={doc_threshold:.0%} similarity")
        trace_steps.append(f"- Found {len(docs)} high-quality document chunks")
        
        if docs:
            for i, doc in enumerate(docs):
                trace_steps.append(f"  {i+1}. {doc['source']} (page {doc['page']}) - {doc['similarity_score']:.1%}")
            
            trace_steps.append("- Action: Using documents for response")
            response = st.session_state.pipeline.step7_generate_response(user_query, [], docs, [], trace)
            end_time = time.time()
            trace_steps.append(f"\n**Response Generation**")
            trace_steps.append(f"- Generated {len(response)} characters in {end_time - start_time:.2f}s")
            trace_data = "\n".join(trace_steps)
            return response, trace_data, end_time - start_time
        
        # Step 6: Web Search
        if config.FALLBACK_CRITERIA["enable_web_fallback"]:
            trace_steps.append("\n**Step 6: Web Search (Final Fallback)**")
            web_results = st.session_state.pipeline.step6_web_search(user_query, trace)
            trace_steps.append(f"- Found {len(web_results)} web results")
            
            if web_results:
                for i, result in enumerate(web_results):
                    trace_steps.append(f"  {i+1}. {result['title']}")
            
            response = st.session_state.pipeline.step7_generate_response(user_query, [], [], web_results, trace)
            end_time = time.time()
            trace_steps.append(f"\n**Response Generation**")
            trace_steps.append(f"- Generated {len(response)} characters in {end_time - start_time:.2f}s")
            trace_data = "\n".join(trace_steps)
            return response, trace_data, end_time - start_time
        else:
            trace_steps.append("\n**Step 6: Web Search**")
            trace_steps.append("- Web fallback disabled")
            end_time = time.time()
            trace_data = "\n".join(trace_steps)
            return "I couldn't find relevant information. Please contact IT support directly.", trace_data, end_time - start_time
    
    except Exception as e:
        end_time = time.time()
        trace_steps.append(f"\n**Error**")
        trace_steps.append(f"- {str(e)}")
        trace_steps.append(f"- Processing time: {end_time - start_time:.2f}s")
        trace_data = "\n".join(trace_steps)
        return "I encountered an error. Please contact IT support directly.", trace_data, end_time - start_time

def main():
    """Main Streamlit application"""
    
    # Clean header
    st.title("IT Helpdesk Assistant")
    st.markdown("Get help with technical issues, software problems, and IT support questions.")
    
    # Initialize
    initialize_session_state()
    
    # Main chat interface
    st.markdown("### Chat")
    
    # Display chat history
    if st.session_state.chat_history:
        for i, (query, response, timestamp, processing_time, trace_data) in enumerate(st.session_state.chat_history):
            # User message
            with st.container():
                st.markdown(f"**You** - {timestamp}")
                st.markdown(f"> {query}")
            
            # Assistant response
            with st.container():
                st.markdown(f"**Assistant** - Response time: {processing_time:.1f}s")
                st.markdown(response)
                
                # Expandable trace
                with st.expander("View Processing Details"):
                    st.markdown(trace_data)
            
            st.divider()
    
    # Query input
    user_query = st.text_input(
        "Ask your IT question:",
        placeholder="e.g., My printer won't connect to WiFi, How do I reset my password?",
        key="user_input"
    )
    
    # Process query
    col1, col2 = st.columns([1, 4])
    with col1:
        send_button = st.button("Send", type="primary", use_container_width=True)
    
    if send_button and user_query:
        
        with st.spinner("Processing..."):
            
            # Process the query with trace
            response, trace_data, processing_time = process_query_simple(user_query)
            
            # Add to chat history
            timestamp = datetime.now().strftime("%H:%M")
            st.session_state.chat_history.append((user_query, response, timestamp, processing_time, trace_data))
            
            # Clear input and rerun
            st.rerun()
    
    # Example queries (simplified)
    if not st.session_state.chat_history:
        st.markdown("### Try these examples:")
        
        examples = [
            "My printer won't connect to WiFi",
            "How do I reset my password?",
            "VPN setup on laptop",
            "Outlook not syncing emails"
        ]
        
        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(example, key=f"example_{i}", use_container_width=True):
                    # Process example query
                    with st.spinner("Processing..."):
                        response, trace_data, processing_time = process_query_simple(example)
                        timestamp = datetime.now().strftime("%H:%M")
                        st.session_state.chat_history.append((example, response, timestamp, processing_time, trace_data))
                        st.rerun()
    
    # Sidebar with minimal info
    with st.sidebar:
        st.markdown("### System Info")
        try:
            ticket_count = st.session_state.data_processor.tickets_collection.count()
            doc_count = st.session_state.data_processor.docs_collection.count()
            st.write(f"Tickets: {ticket_count}")
            st.write(f"Documents: {doc_count}")
        except:
            st.write("System loading...")
        
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":
    main()