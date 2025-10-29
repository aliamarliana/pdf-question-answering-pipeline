"""
LangChain + Pinecone + OpenAI PDF Q&A Pipeline
------------------------------------------------
Loads PDFs, splits them into chunks, embeds them using OpenAI,
stores vectors in Pinecone, and answers user queries.
"""

# ==============================
# Imports
# ==============================
import os
from dotenv import load_dotenv
import pinecone
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as PineconeStore
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain


# ==============================
# Environment Setup
# ==============================
load_dotenv()  # Load API keys and environment variables from .env

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "langchainvector")


# ==============================
# Document Handling
# ==============================
def read_docs(directory: str):
    """Load PDF documents from a directory."""
    loader = PyPDFDirectoryLoader(directory)
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} documents from {directory}")
    return docs


def chunk_docs(docs, chunk_size=800, chunk_overlap=50):
    """Split documents into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks


# ==============================
# Embedding + Pinecone Setup
# ==============================
def setup_embeddings():
    """Initialize OpenAI embeddings."""
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    print("✅ OpenAI Embeddings initialized")
    return embeddings


def setup_pinecone(index_name=INDEX_NAME):
    """Initialize Pinecone connection."""
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    print(f"✅ Pinecone initialized with index: {index_name}")
    return index_name


def create_vectorstore(docs, embeddings, index_name):
    """Store document vectors in Pinecone."""
    index = PineconeStore.from_documents(docs, embeddings, index_name=index_name)
    print("✅ Documents stored in Pinecone vector index")
    return index


# ==============================
# Retrieval + Question Answering
# ==============================
def retrieve_query(index, query, k=2):
    """Retrieve top-k similar documents for a given query."""
    results = index.similarity_search(query, k=k)
    print(f"🔍 Retrieved {len(results)} matching documents")
    return results


def setup_qa_chain():
    """Set up the QA chain using OpenAI LLM."""
    llm = OpenAI(model_name="text-davinci-003", temperature=0.5)
    chain = load_qa_chain(llm, chain_type="stuff")
    print("✅ QA Chain initialized")
    return chain


def get_answer(index, chain, query):
    """Retrieve relevant docs and generate an answer."""
    docs = retrieve_query(index, query)
    answer = chain.run(input_documents=docs, question=query)
    print("\n💬 Answer:")
    print(answer)
    return answer


# ==============================
# Main Execution
# ==============================
if __name__ == "__main__":

    # Step 1: Load and preprocess documents
    docs = read_docs("documents/")
    chunks = chunk_docs(docs)

    # Step 2: Initialize embeddings and vector DB
    embeddings = setup_embeddings()
    index_name = setup_pinecone()
    index = create_vectorstore(chunks, embeddings, index_name)

    # Step 3: Set up LLM QA Chain
    qa_chain = setup_qa_chain()

    # Step 4: Ask a question
    query = "How much the agriculture target will be increased by how many score?"
    get_answer(index, qa_chain, query)
