import os
import json
import PyPDF2
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import re
import requests
import time

class SimpleRAGSystem:
    def __init__(self, model_name="mistral:7b"):
        print("Initializing Improved RAG System...")
        
        self.documents = []
        self.model_name = model_name
        
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"Loaded embedding model: all-MiniLM-L6-v2 (384 dimensions)")
        
        self.vector_store = None
        self.is_fitted = False
        self.embedding_dim = 384
        
        self._check_ollama()
    
    def _check_ollama(self):
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                
                if any(self.model_name in name for name in model_names):
                    print(f"Ollama is running with {self.model_name}")
                else:
                    print(f"Model {self.model_name} not found. Available models: {model_names}")
                    print(f"Run: ollama pull {self.model_name}")
            else:
                print("Ollama is not responding")
        except Exception as e:
            print(f"Ollama connection failed: {e}")
            print("Please start Ollama: ollama serve")

    def create_angelone_sample_data(self):
        """Create proper AngelOne sample data for testing"""
        angelone_data = [
            {
                'source': 'AngelOne Customer Support Guide',
                'content': '''
                AngelOne Customer Support Information
                
                How to Contact Customer Support:
                - Phone: 1800-123-ANGEL (24/7 helpline)
                - Email: support@angelone.in
                - Live Chat: Available on angelone.in website
                - WhatsApp: +91-9876543210
                - Support Hours: Monday to Friday 9:00 AM to 6:00 PM IST
                
                For urgent trading issues during market hours, call our priority line.
                For account opening queries, use live chat for fastest response.
                ''',
                'type': 'angelone_support'
            },
            {
                'source': 'AngelOne Trading Hours & Password Reset',
                'content': '''
                Trading Hours:
                - Equity Market: Monday to Friday 9:15 AM to 3:30 PM IST
                - Currency Market: Monday to Friday 9:00 AM to 5:00 PM IST
                - Commodity Market: Monday to Friday 10:00 AM to 11:30 PM IST
                
                How to Reset Password:
                1. Go to AngelOne login page
                2. Click "Forgot Password" below login button
                3. Enter your registered email address or mobile number
                4. Check your email/SMS for reset link
                5. Click the link and create new password
                6. Password must be 8+ characters with uppercase, lowercase, and numbers
                
                For password reset issues, contact support immediately.
                ''',
                'type': 'angelone_support'
            },
            {
                'source': 'AngelOne Margin Requirements',
                'content': '''
                Margin Requirements:
                
                Delivery Trading:
                - Minimum 20% margin required for delivery trades
                - Full payment required by T+1 day
                
                Intraday Trading:
                - Up to 5x leverage available (20% margin)
                - Positions auto-squared off at 3:20 PM if not closed
                
                F&O Trading:
                - SPAN margin + Exposure margin required
                - Margin call when account falls below 50% of required margin
                - Additional margin may be required for volatile stocks
                
                Margin Calculator available on trading platform for real-time calculations.
                ''',
                'type': 'angelone_support'
            }
        ]
        return angelone_data

    def load_pdf_documents(self, pdf_folder="pdfs"):
        """Load and filter relevant PDF documents"""
        print(f"Loading PDFs from {pdf_folder}/...")
        pdf_documents = []
        
        if not os.path.exists(pdf_folder):
            print(f"PDF folder '{pdf_folder}' not found. Skipping PDF loading.")
            return pdf_documents
        
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
        print(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_folder, pdf_file)
            print(f"Loading: {pdf_file}")
            
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text_content = ""
                    
                    # Only read first few pages to check relevance
                    max_pages = min(3, len(pdf_reader.pages))
                    for page_num in range(max_pages):
                        page = pdf_reader.pages[page_num]
                        text_content += page.extract_text() + "\n"
                    
                    # Check if content is relevant to AngelOne/trading
                    relevant_keywords = ['angelone', 'angel one', 'trading', 'broker', 'margin', 'equity', 'customer support', 'demat', 'securities']
                    content_lower = text_content.lower()
                    
                    if any(keyword in content_lower for keyword in relevant_keywords):
                        # Read full document if relevant
                        full_content = ""
                        for page_num in range(len(pdf_reader.pages)):
                            page = pdf_reader.pages[page_num]
                            full_content += page.extract_text() + "\n"
                        
                        pdf_documents.append({
                            'source': f"PDF: {pdf_file}",
                            'content': full_content.strip(),
                            'type': 'pdf'
                        })
                        print(f"  Loaded relevant PDF: {pdf_file}")
                    else:
                        print(f"  Skipped irrelevant PDF: {pdf_file}")
                        
            except Exception as e:
                print(f"  Error loading {pdf_file}: {str(e)}")
        
        print(f"Loaded {len(pdf_documents)} relevant PDF documents")
        return pdf_documents
    
    def load_scraped_documents(self, scraped_folder="scraped_data"):
        """Load documents from scraped web data"""
        print(f"Loading web data from {scraped_folder}/...")
        web_documents = []
        
        if not os.path.exists(scraped_folder):
            print(f"Scraped data folder '{scraped_folder}' not found. Skipping web data loading.")
            return web_documents
        
        metadata_file = os.path.join(scraped_folder, "metadata.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    scraped_data = json.load(f)
                
                print(f"Found metadata.json with {len(scraped_data)} entries")
                
                for i, page_data in enumerate(scraped_data):
                    if page_data.get('content') and page_data.get('content').strip():
                        web_documents.append({
                            'source': f"Web: {page_data.get('title', f'Page {i+1}')}",
                            'content': page_data['content'],
                            'url': page_data.get('url', ''),
                            'type': 'web'
                        })
                
                print(f"Loaded {len(web_documents)} web documents")
                
            except Exception as e:
                print(f"Error loading metadata: {str(e)}")
        
        return web_documents
    
    def clean_text(self, text):
        """Clean and preprocess text content"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?()-]', ' ', text)
        
        # Remove repeated phrases
        sentences = text.split('.')
        unique_sentences = []
        seen = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 15 and sentence not in seen:
                seen.add(sentence)
                unique_sentences.append(sentence)
        
        return '. '.join(unique_sentences)
    
    def chunk_documents(self, documents, chunk_size=500, overlap=100):
        """Split documents into overlapping chunks"""
        print("Chunking documents...")
        chunked_docs = []
        
        for doc in documents:
            content = self.clean_text(doc['content'])
            
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            
            current_chunk = ""
            chunk_id = 0
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                    if len(current_chunk.strip()) > 100:
                        chunked_docs.append({
                            'source': doc['source'],
                            'content': current_chunk.strip(),
                            'type': doc['type'],
                            'chunk_id': chunk_id
                        })
                    
                    words = current_chunk.split()
                    overlap_words = words[-overlap:] if len(words) > overlap else words
                    current_chunk = ' '.join(overlap_words) + '. ' + sentence
                    chunk_id += 1
                else:
                    current_chunk += '. ' + sentence
            
            if len(current_chunk.strip()) > 100:
                chunked_docs.append({
                    'source': doc['source'],
                    'content': current_chunk.strip(),
                    'type': doc['type'],
                    'chunk_id': chunk_id
                })
        
        print(f"Created {len(chunked_docs)} chunks")
        return chunked_docs
    
    def build_index(self):
        """Build the complete RAG index"""
        print("\nBuilding RAG Index...")
        print("=" * 50)
        
        # Get AngelOne sample data first
        angelone_docs = self.create_angelone_sample_data()
        
        # Load actual documents
        pdf_docs = self.load_pdf_documents()
        web_docs = self.load_scraped_documents()
        
        # Combine all documents
        all_docs = angelone_docs + pdf_docs + web_docs
        
        print(f"\nTotal documents loaded: {len(all_docs)}")
        print(f"   - AngelOne samples: {len(angelone_docs)}")
        print(f"   - PDFs: {len(pdf_docs)}")
        print(f"   - Web pages: {len(web_docs)}")
        
        # Chunk documents
        self.documents = self.chunk_documents(all_docs)
        
        # Create embeddings
        print(f"\nCreating embeddings for {len(self.documents)} chunks...")
        doc_contents = [doc['content'] for doc in self.documents]
        
        embeddings = self.embedding_model.encode(
            doc_contents, 
            show_progress_bar=True,
            batch_size=32
        )
        
        # Build FAISS index
        print("Building FAISS vector store...")
        self.vector_store = faiss.IndexFlatIP(self.embedding_dim)
        
        faiss.normalize_L2(embeddings)
        self.vector_store.add(embeddings.astype('float32'))
        
        self.is_fitted = True
        print("RAG index built successfully!")
        print(f"Index stats:")
        print(f"   - Vector dimension: {self.embedding_dim}")
        print(f"   - Total vectors: {self.vector_store.ntotal}")
        print("=" * 50)
    
    def retrieve_relevant_docs(self, query, top_k=5):
        """Retrieve most relevant documents for a query"""
        if not self.is_fitted:
            raise ValueError("RAG system not fitted! Call build_index() first.")
        
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        similarities, indices = self.vector_store.search(
            query_embedding.astype('float32'), top_k
        )
        
        relevant_docs = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if similarity > 0.15:  # Lower threshold
                doc_content = self.documents[idx]['content']
                doc_source = self.documents[idx]['source']
                
                # Filter out irrelevant documents
                content_lower = doc_content.lower()
                source_lower = doc_source.lower()
                
                skip_terms = ['health insurance', 'america\'s choice', 'medical coverage', 'hsa', 'deductible', 'copay']
                if any(term in content_lower or term in source_lower for term in skip_terms):
                    continue
                
                relevant_docs.append({
                    'content': doc_content,
                    'source': doc_source,
                    'similarity': float(similarity),
                    'type': self.documents[idx]['type']
                })
        
        return relevant_docs
    
    def generate_with_mistral(self, query, context_docs):
        """Generate answer using Ollama Mistral"""
        context = "\n\n".join([
            f"Document: {doc['source']}\nContent: {doc['content']}" 
            for doc in context_docs[:3]
        ])
        
        prompt = f"""You are a customer support assistant for AngelOne, a stock trading platform in India.

Answer the user's question based ONLY on the information provided in the documents below. 
If the information is not available in the documents, say "I Don't know".

DOCUMENTS:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
- Provide a clear, helpful answer based only on the document content
- If the documents don't contain the answer, respond with "I Don't know"
- Be specific and actionable in your response
- Don't make up information not in the documents
- Focus on AngelOne trading platform information

ANSWER:"""
        
        try:
            print("   Calling Mistral...")
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': self.model_name,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.0,
                        'top_p': 0.9,
                        'num_predict': 250
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                answer = response.json()['response'].strip()
                
                if answer.startswith('ANSWER:'):
                    answer = answer[7:].strip()
                
                return answer, context_docs[:3]
            else:
                print(f"Ollama API error: {response.status_code}")
                return self.fallback_answer(query, context_docs)
                
        except Exception as e:
            print(f"Ollama error: {str(e)}")
            return self.fallback_answer(query, context_docs)
    
    def fallback_answer(self, query, relevant_docs):
        """Fallback rule-based answer generation"""
        if not relevant_docs or relevant_docs[0]['similarity'] < 0.3:
            return "I Don't know", []
        
        best_content = relevant_docs[0]['content']
        sentences = [s.strip() for s in best_content.split('.') if s.strip()]
        query_words = set(query.lower().split())
        
        relevant_sentences = []
        for sentence in sentences:
            if len(sentence) > 20:
                sentence_words = set(sentence.lower().split())
                if query_words.intersection(sentence_words):
                    relevant_sentences.append(sentence)
        
        if relevant_sentences:
            return '. '.join(relevant_sentences[:2]) + '.', relevant_docs[:2]
        else:
            return best_content[:300] + '...', relevant_docs[:1]
    
    def query(self, question):
        """Main query method"""
        print(f"\nProcessing query: '{question}'")
        
        if not self.is_fitted:
            return "System not initialized. Please build the index first.", []
        
        print("   Retrieving relevant documents...")
        relevant_docs = self.retrieve_relevant_docs(question, top_k=5)
        
        if not relevant_docs:
            print("   No relevant documents found")
            return "I Don't know", []
        
        print(f"   Found {len(relevant_docs)} relevant documents")
        print(f"   Top similarity: {relevant_docs[0]['similarity']:.3f}")
        
        print("   Generating answer with Mistral...")
        answer, sources = self.generate_with_mistral(question, relevant_docs)
        
        print(f"   Answer generated ({len(answer)} characters)")
        return answer, sources

def main():
    """Main function to run the improved RAG system"""
    print("IMPROVED MISTRAL RAG SYSTEM")
    print("=" * 60)
    
    rag = SimpleRAGSystem(model_name="mistral:7b")
    
    try:
        rag.build_index()
    except Exception as e:
        print(f"Failed to build index: {e}")
        return
    
    print(f"\nRAG System Ready! Type your questions (or 'quit' to exit)")
    print("=" * 60)
    
    while True:
        try:
            question = input("\nYour question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            start_time = time.time()
            answer, sources = rag.query(question)
            end_time = time.time()
            
            print(f"\nANSWER: {answer}")
            print(f"\nSOURCES: {len(sources)} documents used")
            print(f"TIME: {end_time - start_time:.2f} seconds")
            
            if sources:
                print(f"TOP SOURCE: {sources[0]['source']} (similarity: {sources[0]['similarity']:.3f})")
            
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()