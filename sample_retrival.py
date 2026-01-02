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

# yaha pe two ways hai karne ka , either use similarity_search_with_score or as_retriever

# results = vector_store.similarity_search_with_score("who talks about Ankur", k=2)

# print("Results using similarity_search_with_score:")
# for res in results:
#     print(f"* {res[0].page_content} [{res[0].metadata}] -- {res[1]}")


# we can use the above for it , but we shouldnt , instead use as_retriever
# why , cause we have better options to cater our answers accoriding to our needs 
# mtlb , stuff like FCS , lex interpolation etc , usko apan customize kar skyte hai

# yaha k mtlb no of citations chahiye humein

result = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 2 , "score_threshold": 0.6})

response = result.invoke("who does patle say ?")

print("Results using as_retriever:")
for doc in response:
    print(f"* {doc.page_content} [{doc.metadata}]")