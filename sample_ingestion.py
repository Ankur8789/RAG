# sabse pahle basics ko import karo 
import os
import time
from dotenv import load_dotenv
import uuid


# Now pine cone , our own vector DB
from pinecone import Pinecone, ServerlessSpec

# ab langchain ki baari 
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document

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
print("OPENAI_API_KEY:", os.getenv("EMBED_KEY"))


# Embedding vector Model
hf_embeddings = HuggingFaceEndpointEmbeddings(
    model="mixedbread-ai/mxbai-embed-large-v1",  # model on HF Hub
    task="feature-extraction",                    # this tells HF you want embeddings
    huggingfacehub_api_token=os.getenv("EMBED_KEY")  # your token from .env
)


# Setting up our vector store

vector_store = PineconeVectorStore(index = created_index, embedding=hf_embeddings)


# aise hi some documents , using Langchain docs

doc1 = Document(
    page_content="Sandeep said : Hi bolna koi gali toh nahi",
    metadata={"source": "Sandeep"},
)

doc2 = Document(
    page_content="Satyam said : Ankur ka kitna lambda hai",
    metadata={"source": "Satyam"},
)

doc3 = Document(
    page_content="Rqaushan said : Kahe bakchodi kar rha hai",
    metadata={"source": "Raushan"},
)

doc4 = Document(
    page_content="Pratik said : Pagal hai bhai",   
    metadata={"source": "Pratik"},
)

doc5 = Document(
    page_content="Ankur said : Best food is pizza and burger",     
    metadata={"source": "Ankur"},
)

doc6 = Document(
    page_content="Patle said : Pgala lauda",     
    metadata={"source": "Patle"},
)

docs = [doc1, doc2, doc3, doc4, doc5, doc6]

ids = []


# each doc requires unique id and metadata 
for i in range(len(docs)):
    ids.append(f"id{i}")


# storing them into indexed vector store

vector_store.add_documents(documents=docs, ids=ids)
