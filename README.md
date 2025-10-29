# 📘 PDF Question Answering Pipeline

A Python pipeline that uses **LangChain**, **OpenAI**, and **Pinecone** to answer questions from PDF documents.

---

## 🚀 Overview
This project demonstrates how to build a **Retrieval-Augmented Generation (RAG)** pipeline:
- Loads PDFs from a directory  
- Splits them into text chunks  
- Creates embeddings using **OpenAI**  
- Stores vectors in **Pinecone**  
- Retrieves relevant text and answers user queries using **LLMs**

---

## ⚙️ Setup Guide

### 1️⃣ Create a Virtual Environment
```bash
conda create -p venv python==3.10 -y
conda activate ./venv
```


### 2️⃣ Install Dependencies
Create a requirements.txt file (if not already provided) and install:
```bash
pip install -r requirements.txt
```


### 3️⃣ Add Your API Keys
Create a .env file in the project root and include your keys:
```bash
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=gcp-starter
PINECONE_INDEX_NAME=langchainvector
```


### 4️⃣ Add Your Documents
Place all PDF files you want to query inside the ```/documents``` folder:
```bash
project/
│
├── documents/
│   └── report.pdf
├── app.py
├── .env
└── requirements.txt
```


### 5️⃣ Run the Script
```bash
python app.py
```

---

## 💬 Example Query
When prompted or hardcoded in the script:
```bash
query = "How much the agriculture target will be increased by how many score?"
```

You’ll see an answer printed in the console.

---

## 🧠 Tech Stack
- LangChain – document loading, text chunking, and LLM chaining
- OpenAI – text embeddings and question answering
- Pinecone – vector storage and similarity search
- Python-dotenv – environment variable management

---

## 🎥 Code Reference
This project was **inspired by the tutorial by Krish Naik**:  
🔗 [Complete Langchain GEN AI Crash Course With 6 End To End LLM Projects With OPENAI,LLAMA2,Gemini Pro](https://www.youtube.com/watch?v=aWKrL4z5H6w&t=8263s)
