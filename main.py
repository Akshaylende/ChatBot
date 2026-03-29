import os
# Instanting dotenv to fetch environmental variables
from dotenv import load_dotenv
load_dotenv()



# Importing Dependencies
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings




# Pinecone Setup
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Initialize Pinecone client
pc = Pinecone(api_key= os.getenv('PINECONE_API_KEY'))


# Define Index Name
index_name = "langchain-demo"


# Mixtral model info and setup
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0.8,
    api_key=os.getenv("GROQ_API_KEY")
)


# Setting Prompt Templates
from langchain_core.prompts import PromptTemplate

template = """
You are a fortune teller. These Humans will ask you a questions about their life.
Use following piece of context only and not help from other places to answer the question.
If you don't know the answer, just say you don't know and just stick to the horoscope information.
keep the answer within 2 sentences and concise.

Context: {context}
Question: {question}
Answer: 

"""


# Defining ChatBot class
class ChatBot():
    loader = TextLoader('./horoscope.txt')  # File reference for vectorization
    documents = loader.load()

    # text splitting 
    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 4)
    docs = text_splitter.split_documents(documents)

    # Embedding Generations
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Checking Index
    
    # Create a list of all existing index names
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        # Create new Index
        pc.create_index(name = index_name, metric="cosine", dimension = 768)
        docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name = index_name)
    else:
        # Link to the existing index
        docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)


    # setting Final prompt along with context 
    prompt = PromptTemplate(
        template = template,
        input_variables = ["context", "question"]
    )


    