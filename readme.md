# RAG Customer Support Chatbot

A Retrieval-Augmented Generation chatbot that answers customer support questions based on AngelOne documentation.

## 🚀 Live Demo

**Web App:** https://huggingface.co/spaces/adityahegde23/rag-chat-bot

## 📋 Features

- Answers questions based on PDF documents and web scraped data
- Uses semantic search with sentence embeddings
- Powered by Mistral 7B language model
- Returns "I Don't know" for questions outside the knowledge base


### Installation

1. **Clone repository**
   ```bash
   git clone https://huggingface.co/spaces/adityahegde23/rag-chat-bot
   cd rag-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install and run Ollama**
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Start Ollama server
   ollama serve
   
   # Download model (in another terminal)
   ollama pull mistral:7b
   ```

4. **Add your documents**
   - Place PDFs in `pdfs/` folder
   - Place scraped data in `scraped_data/` folder

5. **Run the application**
   ```bash
   python3 app.py
   ```

## 📁 Project Structure

```
rag-chatbot/
├── app.py                 # Gradio web interface
├── rag_system.py          # Main RAG implementation
├── scraper.py             # Web scraping script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── pdfs/                 # PDF documents folder
└── scraped_data/         # Scraped web data folder
```

## 🧪 Testing

Try these example questions:
- "How do I contact customer support?"
- "What are the trading hours?"
- "How do I reset my password?"
- "What is the margin requirement?"

## 🔧 Technical Details

- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Vector Store:** FAISS
- **LLM:** Ollama Mistral 7B
- **UI:** Gradio
- **Document Processing:** PyPDF2, BeautifulSoup

## 📝 Usage

1. Upload your support documents to the appropriate folders
2. Run the application
3. Ask questions through the web interface
4. The system retrieves relevant documents and generates contextual answers
5. If no relevant found, then will return "I Don't know"