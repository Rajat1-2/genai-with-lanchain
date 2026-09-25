from langchain_chroma import Chroma
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
# 3. CREATE CHROMA VECTOR STORE
# ============================================================

vector_store = Chroma(
    collection_name="my_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# ============================================================
# 4. ADD DOCUMENTS
# ============================================================

ids = [
    "doc1",
    "doc2",
    "doc3",
    "doc4"
]

vector_store.add_documents(
    documents=docs,
    ids=ids
)

print("Documents added successfully.")


# ============================================================
# 5. READ ALL DOCUMENTS
# ============================================================

results = vector_store.get(
    include=[
        "documents",
        "metadatas"
    ]
)

print("\n================ ALL DOCUMENTS ================")

print("\nIDs:")
print(results["ids"])

print("\nDocuments:")

for i, document in enumerate(results["documents"]):
    print(f"\nDocument {i + 1}:")
    print(document)

print("\nMetadata:")

for i, metadata in enumerate(results["metadatas"]):
    print(f"\nDocument {i + 1}:")
    print(metadata)


# ============================================================
# 6. SIMILARITY SEARCH
# ============================================================

query = "What is Retrieval Augmented Generation?"

results = vector_store.similarity_search(
    query,
    k=2
)

print("\n================ SIMILARITY SEARCH ================")

for i, doc in enumerate(results):

    print(f"\nResult {i + 1}")

    print("\nContent:")
    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)


# ============================================================
# 7. SIMILARITY SEARCH WITH SCORE
# ============================================================

results_with_score = vector_store.similarity_search_with_score(
    query,
    k=2
)

print("\n================ SIMILARITY SEARCH WITH SCORE ================")

for i, (doc, score) in enumerate(results_with_score):

    print(f"\nResult {i + 1}")

    print("\nContent:")
    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)

    print("\nScore:")
    print(score)


# ============================================================
# 8. SIMILARITY SEARCH WITH METADATA FILTER
# ============================================================

results = vector_store.similarity_search(
    query,
    k=2,
    filter={
        "topic": "Deep Learning"
    }
)

print("\n================ FILTERED SEARCH ================")

for i, doc in enumerate(results):

    print(f"\nResult {i + 1}")

    print("\nContent:")
    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)


# ============================================================
# 9. UPDATE DOCUMENT
# ============================================================

updated_doc = Document(
    page_content="""
    Retrieval-Augmented Generation (RAG) combines a retriever
    with a Large Language Model. The retriever finds relevant
    information from a vector database and provides that
    information to the LLM as context to generate a grounded answer.
    """,
    metadata={
        "source": "rag_updated.txt",
        "topic": "Advanced RAG"
    }
)

# vector_store.update_documents(
#     ids=["doc4"],
#     documents=[updated_doc]
# )

print("\n================ UPDATE ================")

print("Document doc4 updated successfully.")


# ============================================================
# 10. VERIFY UPDATED DOCUMENT
# ============================================================

# updated_results = vector_store.get(
#     ids=["doc4"],
#     include=[
#         "documents",
#         "metadatas"
#     ]
# )

# print("\nUpdated document:")
# print(updated_results["documents"])

# print("\nUpdated metadata:")
# print(updated_results["metadatas"])


# ============================================================
# 11. DELETE DOCUMENT
# ============================================================

vector_store.delete(
    ids=["doc3"]
)

print("\n================ DELETE ================")

print("Document doc3 deleted successfully.")


# ============================================================
# 12. READ FINAL VECTOR STORE
# ============================================================

final_results = vector_store.get(
    include=[
        "documents",
        "metadatas"
    ]
)

print("\n================ FINAL VECTOR STORE ================")

print("\nIDs:")
print(final_results["ids"])

print("\nDocuments:")

for i, document in enumerate(final_results["documents"]):

    print(f"\nDocument {i + 1}:")
    print(document)

print("\nMetadata:")

for i, metadata in enumerate(final_results["metadatas"]):

    print(f"\nDocument {i + 1}:")
    print(metadata)