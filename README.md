# Multilingual-rag-chatbot
A RAG chatbot to chat with pdf and youtube videos.
with memory rentention,meaning this RagBot has Conversational memory which makes it possible to have multiple query search continuously and the Rag application answers based on your previous search query & answers too.


## Flow Diagram of Multilingual RagBot
![Architecture Diagram](ArchitechtureDiag/Flow.jpeg)

## Techstack and Tools used
- Python
- Langchain
- Langraph
- LLama model
- Streamlit
- Google colab
- VS code

## Features
- Upload any PDF and build a searchable knowledge base
- Chunking & embeddings with HuggingFace multilingual embeddings
- Contextual question answering using LangGraph + ChatGroq
- Memory-enabled conversation with context from previous questions
- Streamlit UI for interactive Q&A



## Model used:
- model="llama-3.1-8b-instant"

### API Key:
A Groq API key: set in a .env file as GROQ_API_KEY

### Install dependencies
pip install streamlit langchain-community langchain-groq langchain-core langchain-text-splitters langgraph chromadb python-dotenv
