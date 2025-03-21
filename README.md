# ChatGroq PDF Chatbot
Chat with your documents using the power of **Llama3-8B** and **Groq**! 🚀  
This Streamlit app lets you upload PDFs, converts them into embeddings with HuggingFace, and retrieves smart, contextual answers at lightning speed.


## Requirements

- Python 3.x
- Streamlit – for the web UI
- Langchain - to chain together the retrieval and LLM calls 
- HuggingFace Embeddings – for document vectorization
- FAISS – for efficient similarity search
- Groq + Llama3-8B – for fast, powerful natural language answers
- PyPDFDirectoryLoader – to load PDFs from a directory


## Environment Setup
Before running the app, make sure to set up your environment variables, particularly for the API keys.
1. Create a .env file in the root directory.
2. Add your GROQ_API_KEY in the .env file like so:
   ```bash
   GROQ_API_KEY=<your_groq_api_key>

## How to run
1. Clone the repository:
   ```bash
   git clone https://github.com/AkashR-16/Langchain-Chatbot-FileDirectory-Llama3.git
   cd Langchain-Chatbot-FileDirectory-Llama3
2. Install the dependencies 
   ```bash
   pip install -r requirements.txt
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
4. Open the app in your browser (usually at http://localhost:8501).

## Functionality
- Document Ingestion: The app supports loading PDF files from the ./us_census directory. You can change the directory or add your own documents.
- Document Embedding: Upon clicking the Document Embeddings button, the app processes the documents, generates vector embeddings, and stores them using FAISS for efficient retrieval.
- Question Answering: Users can input their questions related to the documents, and the app will retrieve the most relevant sections of the documents and generate an answer using the Llama3 model.

# 💬 ChatGroq PDF Chatbot

Chat with your documents using the power of **Llama3-8B** and **Groq**! 
This Streamlit app lets you upload PDFs, converts them into embeddings with HuggingFace, and retrieves smart, contextual answers at lightning speed.

---

## 🔁 How It Works (Data Flow)

```text
        ┌─────────────────────────────┐
        │      Start Streamlit App    │
        └────────────┬────────────────┘
                     │
                     ▼
        ┌─────────────────────────────┐
        │ Display UI Title & Prompt   │
        └────────────┬────────────────┘
                     │
                     ▼
     ┌─────────────────────────────────────┐
     │  [Button Click] "Document Embeddings"│
     └────────────────┬────────────────────┘
                      │
                      ▼
     ┌─────────────────────────────────────┐
     │ Load PDFs from "./us_census" folder │
     ├─────────────────────────────────────┤
     │ Split documents into chunks         │
     ├─────────────────────────────────────┤
     │ Create embeddings using HuggingFace │
     ├─────────────────────────────────────┤
     │ Store in FAISS vector DB            │
     └────────────────┬────────────────────┘
                      │
                      ▼
     ┌─────────────────────────────────────┐
     │ [User Input] Enter a question       │
     └────────────────┬────────────────────┘
                      │
                      ▼
     ┌─────────────────────────────────────┐
     │ Retrieve relevant chunks from FAISS │
     ├─────────────────────────────────────┤
     │ Pass context + question to Llama3   │
     ├─────────────────────────────────────┤
     │ Generate answer via ChatGroq        │
     └────────────────┬────────────────────┘
                      │
                      ▼
     ┌─────────────────────────────────────┐
     │ Display answer + response time      │
     ├─────────────────────────────────────┤
     │ Show documents used (if expanded)   │
     └─────────────────────────────────────┘
