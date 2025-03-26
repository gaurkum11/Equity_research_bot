# Equity_research_bot

EquityBot - AI Chatbot for Equity Research Analysis

Overview

EquityBot is a chatbot designed to assist with equity research analysis. It leverages FAISS for vector search, HuggingFace embeddings for text representation, and a retrieval-based question-answering system to provide precise and relevant responses based on available data.

Features

1. Uses FAISS for efficient similarity search.
2. Employs HuggingFace's sentence-transformer model for embeddings.
3. Integrates with a HuggingFace LLM endpoint for response generation.
4. Streamlit-based UI for an interactive chat experience.


## Setup Instructions

1. **Create and Activate a Virtual Environment Using Conda:**  
   ```sh
   conda create --name equitybot python=3.11 -y
   conda activate equitybot


3. Install Dependencies
   pip install -r requirements.txt
   Ensure that your requirements.txt includes:
    streamlit
    langchain
    langchain_community
    langchain_huggingface
    dotenv
    faiss-cpu
    sentence-transformers

3. Set Up Environment Variables
   Create a .env file and add your Hugging Face API token:
   HF_TOKEN=your_huggingface_api_token_here

5. Run the Chatbot
   streamlit run chatbot.py

Significance and Impact :
1. EquityBot enhances financial analysis by: Providing instant responses to equity research queries.
2. Leveraging AI for precise, context-aware insights. 




