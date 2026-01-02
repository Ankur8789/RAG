# sabse pahle basics ko import karo 
import os
import time
from dotenv import load_dotenv
import uuid


# Now pine cone , our own vector DB
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

# ab langchain ki baari 
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader


# load the environment variables from .env file
load_dotenv()

# fetch indexes and stuff from pinecone
ankurPinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Index name kuch for vecrtor store DB
index_name = "le-beta-v2"

# pre check just for validation
existing_indexes = [index_info["name"] for index_info in ankurPinecone.list_indexes()]

if index_name not in existing_indexes:
    ankurPinecone.create_index(
        name=index_name,
        dimension=1024,  # dimension of OpenAI embeddings , generic embeddings
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not ankurPinecone.describe_index(index_name).status["ready"]:
        print("Index is being created, Ruko jara sabar karo ...")
        time.sleep(1)

created_index = ankurPinecone.Index(index_name)
# print("OPENAI_API_KEY:", os.getenv("EMBED_KEY"))


# Embedding vector Model
hf_embeddings = HuggingFaceEndpointEmbeddings(
    model="mixedbread-ai/mxbai-embed-large-v1",  # model on HF Hub
    task="feature-extraction",                    # this tells HF you want embeddings
    huggingfacehub_api_token=os.getenv("EMBED_KEY")  # your token from .env
)


# Setting up our vector store
vector_store = PineconeVectorStore(index = created_index, embedding=hf_embeddings)

# pulling out docs from dosc directory 

loader = PyPDFDirectoryLoader("docs/")

# before doing any further , it is better to change that into Langchain Documents
# By default it is getting changed into Document type (Lang Chain)
raw_docs = loader.load()

# now ab data is not fixed , so custom me kuch v ho skta hai 
# so before making an embedding out of it , make it splitted 

custom_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    keep_separator=True
)

# create docs after splitting 

refined_docs  = custom_text_splitter.split_documents(raw_docs)

ids = []

for i in range(len(refined_docs)):
     ids.append(f'id{i}')

# adding to the data base
vector_store.add_documents(documents=refined_docs, ids=ids)