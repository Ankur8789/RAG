from langchain_huggingface import HuggingFaceEndpointEmbeddings
import streamlit as st
import os
from dotenv import load_dotenv

# Pinecone
from pinecone import Pinecone

# LangChain
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Load environment
load_dotenv()

st.title("Ankur's Chatbot (Groq + Pinecone)")

# ========================
# Pinecone setup
# ========================
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "le-beta-v2"
index = pc.Index(index_name)

# ========================
# Embeddings
# ========================
hf_embeddings = HuggingFaceEndpointEmbeddings(
    model="mixedbread-ai/mxbai-embed-large-v1",
    task="feature-extraction",
    huggingfacehub_api_token=os.getenv("EMBED_KEY"),
)

vector_store = PineconeVectorStore(
    index=index,
    embedding=hf_embeddings
)

# ========================
# Session State
# ========================
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(
            content="You are a helpful assistant that answers using retrieved context."
        )
    ]

# ========================
# Display chat history
# ========================
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# ========================
# User input
# ========================
prompt = st.chat_input("Ask something...")

if prompt:
    # ---- USER MESSAGE ----
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append(HumanMessage(prompt))

    # ========================
    # Retriever
    # ========================
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5},
    )

    print("\n========== SEMANTIC SEARCH ==========")
    print("User Query:", prompt)

    docs = retriever.invoke(prompt)

    print(f"Retrieved {len(docs)} documents")

    # ========================
    # DEBUG OUTPUT (UI)
    # ========================
    with st.expander("🔍 Semantic Search Debug"):
        st.write(f"**Query:** {prompt}")
        st.write(f"**Docs Retrieved:** {len(docs)}")

        for i, doc in enumerate(docs):
            st.markdown(f"### 📄 Document {i+1}")
            st.write("**Content:**")
            st.write(doc.page_content[:1000])
            st.write("**Metadata:**")
            st.json(doc.metadata if doc.metadata else {})

            print(f"\n--- DOCUMENT {i+1} ---")
            print(doc.page_content[:500])
            print("Metadata:", doc.metadata)

    # ========================
    # Context
    # ========================
    context = "\n\n".join(d.page_content for d in docs)

    # ========================
    # System Prompt
    # ========================
    system_prompt = f"""
    You are a question-answering assistant.
    Use the context below to answer the question.
    If the answer is not in the context, say you don't know.
    Answer in at most 3 sentences.

    Context:
    {context}
    """

    print("\n========== SYSTEM PROMPT ==========")
    print(system_prompt)

    st.session_state.messages.append(SystemMessage(system_prompt))

    # ========================
    # Groq LLM
    # ========================
    llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model="openai/gpt-oss-120b",
        temperature=0.7,
    )

    print("\n========== LLM CONFIG ==========")
    print("Model:", "openai/gpt-oss-120b")
    print("Temperature:", 0.7)

    # ========================
    # Invoke LLM
    # ========================
    response = llm.invoke(st.session_state.messages).content

    print("\n========== LLM RESPONSE ==========")
    print(response)

    # ---- ASSISTANT MESSAGE ----
    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append(AIMessage(response))
