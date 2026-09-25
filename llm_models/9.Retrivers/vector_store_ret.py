# 
# from langchain_community.retrievers 

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# ============================================================
# 1. CREATE DOCUMENTS
# ============================================================

docs = [
    Document(
        page_content="""
        Artificial Intelligence is a field of computer science
        that focuses on creating machines capable of performing
        tasks that normally require human intelligence.
        """,
        metadata={
            "source": "ai.txt",
            "topic": "AI"
        }
    ),

    Document(
        page_content="""
        Machine Learning is a subset of Artificial Intelligence.
        It allows computers to learn patterns from data and make
        predictions or decisions without being explicitly programmed.
        """,
        metadata={
            "source": "ml.txt",
            "topic": "Machine Learning"
        }
    ),

    Document(
        page_content="""
        Deep Learning is a subset of Machine Learning that uses
        artificial neural networks with multiple layers. It is
        widely used in computer vision and natural language processing.
        """,
        metadata={
            "source": "dl.txt",
            "topic": "Deep Learning"
        }
    ),

    Document(
        page_content="""
        Retrieval-Augmented Generation combines information retrieval
        with Large Language Models. Relevant documents are retrieved
        from a knowledge base and provided to the LLM as context.
        """,
        metadata={
            "source": "rag.txt",
            "topic": "RAG"
        }
    )
]


# ============================================================
# 2. CREATE EMBEDDING MODEL
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 3. CREATE FAISS VECTOR STORE
# ============================================================

vector_store = FAISS.from_documents(
    documents=docs,
    embedding=embeddings,
    # collection_name="my coll"
)

print("Documents added to FAISS successfully.")


# ============================================================
# 4. CREATE RETRIEVER
# ============================================================

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 2
    }
)


# ============================================================
# 5. QUERY THE RETRIEVER
# ============================================================

query = "What is Retrieval Augmented Generation?"

results = retriever.invoke(query)


# ============================================================
# 6. PRINT RESULTS
# ============================================================

print("\n================ RETRIEVER RESULTS ================")

for i, doc in enumerate(results):

    print(f"\nResult {i + 1}")

    print("\nContent:")
    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)