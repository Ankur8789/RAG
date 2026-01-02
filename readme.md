# 📚 RAG Chatbot (Groq + Pinecone + HuggingFace + Streamlit)

A **Retrieval-Augmented Generation (RAG)** chatbot that lets you chat with your documents using:

- ⚡ **Groq** (fast, FREE LLM inference)
- 🧠 **HuggingFace Inference API** (FREE embeddings)
- 🗂️ **Pinecone** (vector database)
- 🔗 **LangChain** (orchestration)
- 🎨 **Streamlit** (UI)

This project avoids OpenAI entirely and runs fully on **free tiers**.

---

## 🚀 High-Level Architecture

User Query
↓
HuggingFace Embeddings
↓
Pinecone Vector Search
↓
Relevant Chunks
↓
Groq LLM
↓
Final Answer

---

## 🧱 Tech Stack

| Layer        | Technology                                       |
| ------------ | ------------------------------------------------ |
| UI           | Streamlit                                        |
| Embeddings   | HuggingFace `mixedbread-ai/mxbai-embed-large-v1` |
| Vector Store | Pinecone                                         |
| LLM          | Groq (`openai/gpt-oss-120b`)                     |
| Framework    | LangChain                                        |
| Language     | Python 3.10+                                     |

---

## 📁 Project Structure

RAG/
├── chatbot/
│ └── chatbot.py
├── ingestion.py
├── documents/
│ └── \*.pdf
├── venv/
├── .env
├── requirements.txt
└── README.md

---

## 🔐 Environment Variables (`.env`)

Create a `.env` file in the project root:

```env
# Pinecone
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=le-beta-v2

# Groq
GROQ_API_KEY=your_groq_api_key

# HuggingFace (Embeddings)
EMBED_KEY=your_huggingface_read_token


📦 Install Dependencies
pip install -r requirements.txt

requirements.txt
streamlit
python-dotenv
pinecone-client
langchain
langchain-core
langchain-community
langchain-pinecone
langchain-groq
langchain-huggingface

🌐 Service Setup
1️⃣ Pinecone Setup

Go to https://www.pinecone.io

Create a Serverless Index

Set:

Dimension: 1024

Metric: cosine

Save the index name in .env

⚠️ Dimension must match mxbai-embed-large-v1

2️⃣ Groq Setup (FREE LLM)

Go to https://console.groq.com

Create an API key

Supported models:

✅ openai/gpt-oss-120b

✅ llama3-8b-8192

❌ llama3-70b-8192 (deprecated)

3️⃣ HuggingFace Setup (FREE Embeddings)

Go to https://huggingface.co/settings/tokens

Create a Read-only token

Use it as EMBED_KEY

📥 Document Ingestion Flow
PDFs
 ↓
PyPDFDirectoryLoader
 ↓
LangChain Documents
 ↓
Text Splitter
 ↓
Embeddings
 ↓
Pinecone

Correct Import (LangChain v0.2+)
from langchain_community.document_loaders import PyPDFDirectoryLoader

▶️ Run the Chatbot
streamlit run chatbot/chatbot.py

🧠 RAG Prompt Strategy

The LLM is instructed to:

Use only retrieved context

Say "I don't know" if answer isn’t in context

Answer in ≤ 3 sentences

🔍 Debugging & Observability

The app prints:

🔎 User query

📄 Retrieved documents

📊 Similarity scores

🧠 Final system prompt

🤖 LLM response

This helps debug semantic search quality.
```
