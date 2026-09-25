from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq

from langchain_core.documents import Document

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set")


# --------------------------------
# 1. Documents
# --------------------------------

docs = [
    Document(
        page_content="""
        Machine learning is a branch of artificial intelligence.
        It allows computers to learn patterns from data.

        Supervised learning uses labeled training data.
        The model learns a relationship between input and output.

        Machine learning is used in healthcare, finance,
        recommendation systems and robotics.

        Python is commonly used for machine learning.
        """,
        metadata={"topic": "machine-learning"}
    ),

    Document(
        page_content="""
        Deep learning is a subset of machine learning.
        It uses neural networks with multiple layers.

        Deep learning is commonly used in computer vision,
        natural language processing and speech recognition.
        """,
        metadata={"topic": "deep-learning"}
    )
]


# --------------------------------
# 2. Embeddings
# --------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------
# 3. Vector Store
# --------------------------------

vector_store = FAISS.from_documents(
    docs,
    embeddings
)


# --------------------------------
# 4. Base Retriever
# --------------------------------

base_retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)


# --------------------------------
# 5. LLM
# --------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=groq_api_key,
    temperature=0
)


# --------------------------------
# 6. Compressor
# --------------------------------

compressor = LLMChainExtractor.from_llm(llm)


# --------------------------------
# 7. Contextual Compression Retriever
# --------------------------------

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)


# --------------------------------
# 8. Query
# --------------------------------

query = "What is supervised learning?"


# --------------------------------
# 9. Retrieve compressed documents
# --------------------------------

results = compression_retriever.invoke(query)


# --------------------------------
# 10. Print results
# --------------------------------

for i, doc in enumerate(results, 1):

    print(f"\n--- Document {i} ---")

    print("Content:")
    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)