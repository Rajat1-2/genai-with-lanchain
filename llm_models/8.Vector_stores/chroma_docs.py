from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# 1. CREATE DOCUMENTS
# ============================================================

docs = [
    {
        "page_content": """
        Artificial Intelligence is a field of computer science
        that focuses on creating machines capable of performing
        tasks that normally require human intelligence.
        """,
        "metadata": {
            "source": "ai.txt",
            "topic": "AI"
        }
    },

    {
        "page_content": """
        Machine Learning is a subset of Artificial Intelligence.
        It allows computers to learn patterns from data and make
        predictions or decisions without being explicitly programmed.
        """,
        "metadata": {
            "source": "ml.txt",
            "topic": "Machine Learning"
        }
    },

    {
        "page_content": """
        Deep Learning is a subset of Machine Learning that uses
        artificial neural networks with multiple layers. It is
        widely used in computer vision and natural language processing.
        """,
        "metadata": {
            "source": "dl.txt",
            "topic": "Deep Learning"
        }
    },

    {
        "page_content": """
        Retrieval-Augmented Generation combines information retrieval
        with Large Language Models. Relevant documents are retrieved
        from a knowledge base and provided to the LLM as context.
        """,
        "metadata": {
            "source": "rag.txt",
            "topic": "RAG"
        }
    }
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

ids = ["doc1", "doc2", "doc3", "doc4"]

# vector_store.add_texts(
#     texts=[doc["page_content"] for doc in docs],
#     metadatas=[doc["metadata"] for doc in docs],
#     ids=ids
# )
vector_store.add_documents(docs)

print("Documents added successfully.")


# ============================================================
# 5. READ ALL DOCUMENTS
# ============================================================

results = vector_store.get(include=['embeddings','documents','metadata'])

print("\n================ ALL DOCUMENTS ================")

print("IDs:")
print(results["ids"])

print("\nDocuments:")
print(results["documents"])

print("\nMetadata:")
print(results["metadatas"])


# ============================================================
# 6. SIMILARITY SEARCH
# ============================================================

query = "What is retrieval augmented generation?"

results = vector_store.similarity_search(
    query,
    k=2
)

res=vector_store.similarity_search_with_score(
    query,
    filter={"topic":"deep learning"}
)

print("\n================ SIMILARITY SEARCH ================")

for i, doc in enumerate(results):

    print(f"\nResult {i + 1}")

    print("Content:")
    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)


# metadata
print("\n================ SIMILARITY SEARCH with metadata ================")
for i, doc in enumerate(results):

    print(f"\nResult {i + 1}")

    print("Content:")
    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)



# ============================================================
# 7. UPDATE DOCUMENT
# ============================================================

# vector_store.update_documents(
#     ids=["doc4"],
#     documents=[
#         {
#             "page_content": """
#             Retrieval-Augmented Generation (RAG) combines a retriever
#             with a Large Language Model. The retriever finds relevant
#             information from a vector database and the LLM uses that
#             information as context to generate a grounded answer.
#             """,
#             "metadata": {
#                 "source": "rag_updated.txt",
#                 "topic": "Advanced RAG"
#             }
#         }
#     ]
# )

print("\nDocument updated successfully.")


# ============================================================
# 8. DELETE DOCUMENT
# ============================================================

vector_store.delete(ids=["doc3"])

print("Document doc3 deleted successfully.")


# ============================================================
# 9. READ AGAIN AFTER UPDATE + DELETE
# ============================================================

results = vector_store.get()

print("\n================ FINAL DOCUMENTS ================")

print("IDs:")
print(results["ids"])

print("\nDocuments:")
print(results["documents"])

print("\nMetadata:")
print(results["metadatas"])