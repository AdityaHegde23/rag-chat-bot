"""
Simple Gradio UI for RAG Customer Support Chatbot
Clean and minimal interface
"""

import gradio as gr
import time

from rag_system import SimpleRAGSystem  

class SimpleChatbotUI:
    def __init__(self):
        self.rag_system = SimpleRAGSystem(model_name="mistral:7b") 
        self.is_initialized = False
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize the RAG system"""
        try:
            print("Initializing RAG system...")
            self.rag_system.build_index()
            self.is_initialized = True
            print("RAG system ready!")
        except Exception as e:
            print(f"Error initializing RAG system: {str(e)}")
            self.is_initialized = False
    
    def get_response(self, message, history):
        """Generate chatbot response"""
        if not self.is_initialized:
            return "System not initialized properly. Please check the setup."
        
        if not message.strip():
            return "Please ask me something!"
        
        try:
            answer, sources = self.rag_system.query(message)
            
            response = answer
            
            if sources:
                response += "\n\n**Sources:**\n"
                for i, source in enumerate(sources[:3], 1):
                    similarity = source.get('similarity', 0)
                    response += f"{i}. {source['source']} (relevance: {similarity:.2f})\n"
            
            return response
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def create_interface(self):
        """Create simple Gradio interface"""
        
        with gr.Blocks(title="RAG Chatbot") as interface:
            
            gr.Markdown("# RAG Chatbot")
            
            # Chat interface
            chatbot = gr.Chatbot(height=400)
            
            # Input
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask me anything...",
                    label="Your Question",
                    scale=4
                )
                send_btn = gr.Button("Send", scale=1)
            
            # Example questions
            gr.Markdown("### Example Questions:")
            
            examples = [
                "How do I contact customer support?",
                "What are the trading hours?",
                "How do I reset my password?",
                "What is the margin requirement?"
            ]
            
            for example in examples:
                gr.Button(example, size="sm").click(
                    lambda x=example: x, outputs=msg
                )
            
            # Event handlers
            def respond(message, chat_history):
                if not message.strip():
                    return chat_history, ""
                
                # Add user message
                chat_history = chat_history + [[message, None]]
                
                # Get bot response
                bot_response = self.get_response(message, chat_history)
                
                # Add bot response
                chat_history[-1][1] = bot_response
                
                return chat_history, ""
            
            # Connect events
            send_btn.click(respond, [msg, chatbot], [chatbot, msg])
            msg.submit(respond, [msg, chatbot], [chatbot, msg])
        
        return interface

def main():
    """Run the chatbot"""
    print("Starting RAG Chatbot...")
    
    # Create UI
    ui = SimpleChatbotUI()
    
    # Create interface
    interface = ui.create_interface()
    
    # Launch
    print("Launching interface...")
    interface.launch(
        share=True,
        server_port=7860
    )

if __name__ == "__main__":
    main()