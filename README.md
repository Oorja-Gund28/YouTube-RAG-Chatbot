# YouTube-RAG-Chatbot
An AI-powered chatbot that enables users to interact with YouTube videos using natural language. The application leverages Retrieval-Augmented Generation (RAG) and LangChain to extract video transcripts, perform semantic search, and generate context-aware responses using Large Language Models.

### 🚀 Features:
Extracts transcripts directly from YouTube videos.
Converts transcripts into vector embeddings.
Stores embeddings in a vector database.
Performs semantic similarity search.
Generates context-aware answers using LLMs.
Reduces hallucinations through Retrieval-Augmented Generation (RAG).
Interactive conversational interface.

### 🛠️ Tech Stack:
Python,
LangChain,
Google Gemini,
FAISS, 
YouTube Transcript API,
Gemini Embeddings,

### 📊 Architecture:
YouTube Video URL
        │
        ▼
Transcript Extraction
        │
        ▼
Text Chunking
        │
        ▼
Embedding Generation
        │
        ▼
Vector Database
        │
        ▼
Retriever
        │
        ▼
LLM (Gemini/OpenAI)
        │
        ▼
Context-Aware Answer
