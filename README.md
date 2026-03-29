# RAG - ChatBot

## Project Overview
Although Large Language Models (LLMs) are powerful and capable of generating content, they can produce outdated, generic or incorrect information as they are trained on static data. To overcome this limitation, Retrieval Augmented Generation (RAG) systems can be used to connect the LLM to external data and obtain more reliable answers.

The aim of this project is to build a RAG chatbot in Langchain powered by pinecone's, Hugging Face and Groq APIs. You can upload documents in txt, pdf, CSV, or docx formats and chat with your data. Relevant documents will be retrieved and sent to the LLM along with your follow-up questions for accurate answers.

Throughout this project, we examined each component of the RAG system from document loader to conversational retrieval chain. Additionally, we are planning to develop a user interface using streamlit application.


## Installation
This project requires mainly Python 3 and the following Python libraries installed:

langchain, HuggingFace, Pinecone, Chatgroq, streamlit

The full list of requirements can be found in requirements.txt


## Instructions
To run the app locally:
1. Create a virtual environment: python -m venv venv
2. Activate the virtual environment : .\venv\Scripts\activate (windows)
3. Run the following command in the directory: cd chatbot
4. Install the required dependencies pip install -r requirements.txt
5. Run the following command in the current directory: python main.py
6. Chat with your documents: ask questions and get 🤖 AI answers.

## Flow Diagram && Working

![alt text](c:/Users/aksha/Downloads/1_KyeKTHiXGKoZTpE0PllH2w.webp)