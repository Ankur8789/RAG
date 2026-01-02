# firse basics import karo 
import os
from dotenv import load_dotenv

from pinecone import Pinecone,ServerlessSpec

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document

# load environment variables from .env file
load_dotenv()

# mera pinecone setup 
ankurPinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "le-beta-v2"

# get the index
index = ankurPinecone.Index(index_name)

# embedding model
hf_embeddings = HuggingFaceEndpointEmbeddings(model="mixedbread-ai/mxbai-embed-large-v1",
                                              task="feature-extraction",
                                              huggingfacehub_api_token=os.getenv("EMBED_KEY"))
# vector store setup
vector_store = PineconeVectorStore(index=index, embedding=hf_embeddings)


# retrive karte hai sample data se 
# yaha k mtlb no of citations chahiye humein

result = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 2 , "score_threshold": 0.6})

response = result.invoke("who has ankur done in his work")

print("Results using as_retriever:")
for doc in response:
    print(f"* {doc.page_content} [{doc.metadata}]")