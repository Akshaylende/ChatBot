import os
# Instanting dotenv to fetch environmental variables
from dotenv import load_dotenv
load_dotenv()




# Importing Dependencies
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings





# Defining ChatBot class
class ChatBot():
    loader = TextLoader('')  # File reference for vectorization
    documents = loader.load()

    # text splitting 
    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 4)
    docs = text_splitter.split_documents(documents)

    # Embedding Generations
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    