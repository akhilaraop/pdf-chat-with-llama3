# Langchain-Chatbot-FileDirectory-Llama3
This project demonstrates the use of `ChatGroq` with `Llama3-8b-8192` for building an interactive document-based question answering system using Streamlit. The app allows users to load documents (in this case, PDF files), generate vector embeddings for efficient retrieval, and answer questions based on the loaded documents.

## Requirements

- Python 3.x
- Streamlit
- Langchain
- FAISS
- OllamaEmbeddings
- dotenv

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
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
4. Open the app in your browser (usually at http://localhost:8501).

## Functionality
- Document Ingestion: The app supports loading PDF files from the ./us_census directory. You can change the directory or add your own documents.
- Document Embedding: Upon clicking the Document Embeddings button, the app processes the documents, generates vector embeddings, and stores them using FAISS for efficient retrieval.
- Question Answering: Users can input their questions related to the documents, and the app will retrieve the most relevant sections of the documents and generate an answer using the Llama3 model.
