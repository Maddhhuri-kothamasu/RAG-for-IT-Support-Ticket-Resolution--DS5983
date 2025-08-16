# Main pipeline for IT Helpdesk RAG Chatbot

import json
import logging
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import openai
from ddgs import DDGS

from config import config, validate_config
from prompts import PromptTemplates
from data_processor import DataProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PipelineTrace:
    """Track the execution trace for debugging and UI display"""
    original_query: str = ""
    classification_result: Dict[str, Any] = None
    expanded_query: str = ""
    tickets_found: List[Dict] = None
    ticket_relevance: Dict[str, Any] = None
    docs_found: List[Dict] = None
    web_results: List[Dict] = None
    fallback_triggered: str = ""  # "docs", "web", or ""
    final_prompt: str = ""
    final_response: str = ""
    error_message: str = ""

class ITHelpdeskPipeline:
    def __init__(self):
        validate_config()
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        self.data_processor = DataProcessor()
        self.prompts = PromptTemplates()
        
    def _call_openai(self, prompt: str, temperature: float = 0.3, response_format: Dict = None) -> str:
        """Make OpenAI API call with error handling"""
        try:
            kwargs = {
                "model": config.OPENAI_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature
            }
            
            if response_format:
                kwargs["response_format"] = response_format
                
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def step1_classify_query(self, query: str, trace: PipelineTrace) -> bool:
        """Step 1: Check if query is IT-related"""
        logger.info("Step 1: Classifying query")
        
        prompt = self.prompts.get_classification_prompt().format(query=query)
        
        try:
            response = self._call_openai(
                prompt, 
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response)
            trace.classification_result = result
            
            logger.info(f"Classification: {result}")
            return result.get("is_it_query", False)
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            trace.error_message = f"Classification failed: {e}"
            return False

    def step2_expand_query(self, query: str, trace: PipelineTrace) -> str:
        """Step 2: Expand query with synonyms and related terms"""
        logger.info("Step 2: Expanding query")
        
        prompt = self.prompts.get_query_expansion_prompt().format(query=query)
        
        try:
            expanded = self._call_openai(prompt, temperature=0.5)
            trace.expanded_query = expanded.strip()
            
            logger.info(f"Expanded query: {expanded}")
            return expanded.strip()
            
        except Exception as e:
            logger.error(f"Query expansion error: {e}")
            trace.error_message = f"Query expansion failed: {e}"
            return query  # Fallback to original query

    def step3_search_tickets(self, query: str, trace: PipelineTrace) -> List[Dict]:
        """Step 3: Search for similar tickets with quality filtering"""
        logger.info("Step 3: Searching tickets with quality threshold")
        
        try:
            # Get more results to filter from
            results = self.data_processor.search_tickets(
                query, 
                config.FALLBACK_CRITERIA["max_tickets_to_check"]
            )
            
            tickets = []
            similarity_threshold = config.FALLBACK_CRITERIA["ticket_similarity_threshold"]
            
            for i, (doc, meta, distance) in enumerate(zip(
                results["documents"], 
                results["metadatas"], 
                results["distances"]
            )):
                similarity_score = 1 - distance  # Convert distance to similarity
                
                # Only include tickets above similarity threshold
                if similarity_score >= similarity_threshold:
                    ticket = {
                        "rank": len(tickets) + 1,  # Rank among filtered tickets
                        "id": meta.get("id", ""),
                        "subject": meta.get("subject", ""),
                        "content": doc,
                        "resolution": meta.get("resolution_strategy", ""),
                        "topic": meta.get("inferred_topic", ""),
                        "category": meta.get("category", ""),
                        "priority": meta.get("priority", ""),
                        "similarity_score": similarity_score
                    }
                    tickets.append(ticket)
                
            trace.tickets_found = tickets
            logger.info(f"Found {len(tickets)} high-quality tickets (≥{similarity_threshold:.0%} similarity)")
            return tickets
            
        except Exception as e:
            logger.error(f"Ticket search error: {e}")
            trace.error_message = f"Ticket search failed: {e}"
            return []

    def step4_check_relevance(self, query: str, tickets: List[Dict], trace: PipelineTrace) -> Tuple[bool, List[Dict]]:
        """Step 4: Check if high-quality tickets are relevant"""
        logger.info("Step 4: Checking ticket relevance")
        
        if not tickets:
            trace.ticket_relevance = {"has_useful_tickets": False, "recommendation": "try_documents"}
            logger.info("No high-quality tickets found, will try documents")
            return False, []
            
        # All tickets are already above similarity threshold, now check AI relevance
        tickets_text = ""
        for ticket in tickets:  # Check all filtered tickets
            tickets_text += f"""
Ticket ID: {ticket['id']}
Subject: {ticket['subject']}
Resolution: {ticket['resolution']}
Topic: {ticket['topic']}
Category: {ticket['category']}
Similarity Score: {ticket['similarity_score']:.3f}
---
"""
        
        prompt = self.prompts.get_relevance_check_prompt().format(
            query=query,
            tickets=tickets_text
        )
        
        try:
            response = self._call_openai(
                prompt,
                response_format={"type": "json_object"}
            )
            
            relevance = json.loads(response)
            trace.ticket_relevance = relevance
            
            logger.info(f"Relevance check: {relevance}")
            
            # If we have high-quality tickets and AI confirms relevance, use them
            has_useful = relevance.get("has_useful_tickets", False)
            overall_score = relevance.get("overall_relevance", 0.0)
            
            # Since tickets already meet similarity threshold, be more lenient with AI check
            is_relevant = (
                has_useful and 
                overall_score >= config.FALLBACK_CRITERIA["relevance_threshold"]
            )
            
            if is_relevant:
                logger.info(f"Using {len(tickets)} high-quality tickets for response")
            else:
                logger.info("High-quality tickets found but not contextually relevant, trying documents")
            
            return is_relevant, tickets
            
        except Exception as e:
            logger.error(f"Relevance check error: {e}")
            trace.error_message = f"Relevance check failed: {e}"
            return False, tickets

    def step5_search_documents(self, query: str, trace: PipelineTrace) -> List[Dict]:
        """Step 5: Search documentation with quality filtering"""
        logger.info("Step 5: Searching documents with quality threshold")
        trace.fallback_triggered = "docs"
        
        try:
            # Get more results to filter from  
            results = self.data_processor.search_documents(
                query, 
                config.FALLBACK_CRITERIA["max_docs_to_check"]
            )
            
            docs = []
            similarity_threshold = config.FALLBACK_CRITERIA["doc_similarity_threshold"]
            
            for i, (doc, meta, distance) in enumerate(zip(
                results["documents"],
                results["metadatas"], 
                results["distances"]
            )):
                similarity_score = 1 - distance
                
                # Only include documents above similarity threshold
                if similarity_score >= similarity_threshold:
                    doc_chunk = {
                        "rank": len(docs) + 1,  # Rank among filtered docs
                        "source": meta.get("source", ""),
                        "page": meta.get("page", ""),
                        "content": doc,
                        "similarity_score": similarity_score
                    }
                    docs.append(doc_chunk)
                
            trace.docs_found = docs
            logger.info(f"Found {len(docs)} high-quality document chunks (≥{similarity_threshold:.0%} similarity)")
            return docs
            
        except Exception as e:
            logger.error(f"Document search error: {e}")
            trace.error_message = f"Document search failed: {e}"
            return []

    def step6_web_search(self, query: str, trace: PipelineTrace) -> List[Dict]:
        """Step 6: Web search fallback if local data insufficient"""
        if not config.FALLBACK_CRITERIA["enable_web_fallback"]:
            return []
            
        logger.info("Step 6: Web search fallback")
        trace.fallback_triggered = "web"
        
        try:
            ddgs = DDGS()
            search_query = f"{query} IT support troubleshooting"
            results = list(ddgs.text(
                search_query, 
                max_results=config.FALLBACK_CRITERIA["web_search_queries"],
                region='us-en',
                safesearch='moderate'
            ))
            
            web_results = []
            for i, result in enumerate(results):
                web_result = {
                    "rank": i + 1,
                    "title": result.get("title", ""),
                    "body": result.get("body", ""),
                    "url": result.get("href", "")
                }
                web_results.append(web_result)
                
            trace.web_results = web_results
            logger.info(f"Found {len(web_results)} web results")
            return web_results
            
        except Exception as e:
            logger.error(f"Web search error: {e}")
            trace.error_message = f"Web search failed: {e}"
            return []

    def step7_generate_response(self, query: str, tickets: List[Dict], docs: List[Dict], 
                              web_results: List[Dict], trace: PipelineTrace) -> str:
        """Step 7: Generate final response"""
        logger.info("Step 7: Generating response")
        
        try:
            if web_results:
                # Use web search results
                web_text = ""
                for result in web_results:
                    web_text += f"Title: {result['title']}\nContent: {result['body']}\nURL: {result['url']}\n\n"
                
                internal_context = ""
                if tickets:
                    internal_context += "Internal Tickets:\n"
                    for ticket in tickets[:3]:
                        internal_context += f"- {ticket['subject']}: {ticket['resolution']}\n"
                        
                prompt = self.prompts.get_final_generation_prompt_web().format(
                    query=query,
                    web_results=web_text,
                    internal_context=internal_context
                )
                
            elif tickets or docs:
                # Use local data
                tickets_text = ""
                if tickets:
                    for ticket in tickets[:5]:
                        tickets_text += f"""
Ticket: {ticket['subject']}
Problem: {ticket['content'][:200]}...
Solution: {ticket['resolution']}
Topic: {ticket['topic']} | Category: {ticket['category']}
---
"""
                
                docs_text = ""
                if docs:
                    for doc in docs[:3]:
                        docs_text += f"""
Source: {doc['source']} (Page {doc['page']})
Content: {doc['content'][:300]}...
---
"""
                
                prompt = self.prompts.get_final_generation_prompt_local().format(
                    query=query,
                    tickets=tickets_text,
                    docs=docs_text
                )
                
            else:
                # No useful information found
                prompt = f"""User asked: "{query}"

No relevant information was found in our knowledge base. Provide a helpful response suggesting they contact IT support directly and mention what information they should provide when contacting support."""
            
            trace.final_prompt = prompt
            response = self._call_openai(prompt, temperature=0.7)
            trace.final_response = response
            
            logger.info("Response generated successfully")
            return response
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            trace.error_message = f"Response generation failed: {e}"
            return "I apologize, but I encountered an error while generating a response. Please contact IT support directly for assistance."

    def process_query(self, user_query: str) -> Tuple[str, PipelineTrace]:
        """Main pipeline processing function"""
        trace = PipelineTrace(original_query=user_query)
        
        try:
            # Step 1: Classify query
            is_it_query = self.step1_classify_query(user_query, trace)
            if not is_it_query:
                response = self.prompts.get_non_it_response()
                trace.final_response = response
                return response, trace
            
            # Step 2: Expand query
            expanded_query = self.step2_expand_query(user_query, trace)
            
            # Step 3: Search tickets
            tickets = self.step3_search_tickets(expanded_query, trace)
            
            # Step 4: Check relevance
            tickets_relevant, relevant_tickets = self.step4_check_relevance(user_query, tickets, trace)
            
            if tickets_relevant:
                # Use high-quality tickets for response
                response = self.step7_generate_response(user_query, relevant_tickets, [], [], trace)
                return response, trace
            
            # Step 5: Search documents (fallback) - only if no good tickets
            docs = self.step5_search_documents(expanded_query, trace)
            
            if len(docs) > 0:  # Any high-quality docs found
                # Use high-quality documents for response
                response = self.step7_generate_response(user_query, [], docs, [], trace)
                return response, trace
            
            # Step 6: Web search (final fallback) - only if no good local data
            web_results = self.step6_web_search(user_query, trace)
            
            # Step 7: Generate final response
            response = self.step7_generate_response(
                user_query, relevant_tickets, docs, web_results, trace
            )
            
            return response, trace
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            trace.error_message = f"Pipeline failed: {e}"
            error_response = "I apologize, but I encountered an unexpected error. Please contact IT support directly."
            trace.final_response = error_response
            return error_response, trace

def main():
    """Test the pipeline"""
    pipeline = ITHelpdeskPipeline()
    
    test_queries = [
        "My printer won't connect to WiFi",
        "How do I reset my password?",
        "What's the weather like today?",
        "VPN is not working on my laptop"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Testing query: {query}")
        print('='*60)
        
        response, trace = pipeline.process_query(query)
        
        print(f"Final Response: {response}")
        print(f"Fallback triggered: {trace.fallback_triggered}")
        if trace.error_message:
            print(f"Errors: {trace.error_message}")

if __name__ == "__main__":
    main()
