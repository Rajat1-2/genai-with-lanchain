from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq

from langchain_core.documents import Document

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
# from langchain_community.mul


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set")


# ============================================================
# 2. CREATE DOCUMENTS
# ============================================================

docs = [

    Document(
        page_content="""
        Machine Learning is a branch of Artificial Intelligence
        that allows computers to learn patterns from data and make
        predictions without being explicitly programmed.
        """,
        metadata={
            "source": "ml_1.txt",
            "topic": "Machine Learning"
        }
    ),

    Document(
        page_content="""
        Machine Learning algorithms learn from historical data.
        The learned patterns can then be used to make predictions
        about new and unseen data.
        """,
        metadata={
            "source": "ml_2.txt",
            "topic": "Machine Learning"
        }
    ),

    Document(
        page_content="""
        Supervised Learning is a type of Machine Learning where
        models are trained using labeled data. Classification and
        regression are common supervised learning tasks.
        """,
        metadata={
            "source": "supervised.txt",
            "topic": "Supervised Learning"
        }
    ),

    Document(
        page_content="""
        Unsupervised Learning works with unlabeled data. The model
        tries to discover hidden patterns or structures in the data.
        Clustering is a common unsupervised learning technique.
        """,
        metadata={
            "source": "unsupervised.txt",
            "topic": "Unsupervised Learning"
        }
    ),

    Document(
        page_content="""
        Deep Learning is a subset of Machine Learning that uses
        multi-layer neural networks. It is widely used in computer
        vision, speech recognition, and natural language processing.
        """,
        metadata={
            "source": "deep_learning.txt",
            "topic": "Deep Learning"
        }
    ),

    Document(
        page_content="""
        Reinforcement Learning is a Machine Learning technique where
        an agent learns by interacting with an environment and receiving
        rewards or penalties for its actions.
        """,
        metadata={
            "source": "reinforcement.txt",
            "topic": "Reinforcement Learning"
        }
    )
]


# ============================================================
# 3. CREATE EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 4. CREATE FAISS VECTOR STORE
# ============================================================

vector_store = FAISS.from_documents(
    documents=docs,
    embedding=embeddings
)

print("Vector store created successfully.")


# ============================================================
# 5. CREATE GROQ LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=groq_api_key,
    temperature=0
)


# ============================================================
# 6. CREATE MULTI QUERY RETRIEVER
# ============================================================

retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(
        search_kwargs={
            "k": 2
        }
    ),
    llm=llm
)


# ============================================================
# 7. QUERY
# ============================================================

query = "What is machine learning?"


# ============================================================
# 8. RETRIEVE
# ============================================================

results = retriever.invoke(query)


# ============================================================
# 9. PRINT RESULTS
# ============================================================

print("\n================ MULTI QUERY RESULTS ================")

print("Number of documents:", len(results))

for i, doc in enumerate(results):

    print(f"\n========== RESULT {i + 1} ==========")

    print("\nContent:")
    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)