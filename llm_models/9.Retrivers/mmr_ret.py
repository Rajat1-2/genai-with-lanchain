from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# --------------------------------------------------
# 1. Load documents
# --------------------------------------------------
from langchain_core.documents import Document

docs = [

    Document(
        page_content="""
        Machine Learning is a branch of Artificial Intelligence that
        allows computers to learn patterns from data and make predictions
        without being explicitly programmed.
        """,
        metadata={"source": "ml_1.txt", "topic": "Machine Learning"}
    ),

    Document(
        page_content="""
        Machine Learning is a subset of Artificial Intelligence in which
        computers learn patterns from data and use those patterns to make
        predictions or decisions automatically.
        """,
        metadata={"source": "ml_2.txt", "topic": "Machine Learning"}
    ),

    Document(
        page_content="""
        Machine Learning enables computers to learn from data instead of
        being explicitly programmed. The system identifies patterns in
        training data and uses them to make predictions.
        """,
        metadata={"source": "ml_3.txt", "topic": "Machine Learning"}
    ),

    Document(
        page_content="""
        Machine learning algorithms learn from historical data and use
        the learned patterns to predict outcomes for new data. It is an
        important area of Artificial Intelligence.
        """,
        metadata={"source": "ml_4.txt", "topic": "Machine Learning"}
    ),

    Document(
        page_content="""
        Supervised learning is a type of Machine Learning where models
        learn from labeled training examples. Classification and regression
        are common supervised learning tasks.
        """,
        metadata={"source": "supervised.txt", "topic": "Supervised Learning"}
    ),

    Document(
        page_content="""
        Unsupervised learning is a Machine Learning approach where the
        algorithm works with unlabeled data. Clustering is one of the
        most common unsupervised learning techniques.
        """,
        metadata={"source": "unsupervised.txt", "topic": "Unsupervised Learning"}
    ),

    Document(
        page_content="""
        Deep Learning is a specialized area of Machine Learning that uses
        neural networks containing multiple layers. It is commonly used
        for image recognition, speech processing, and natural language
        processing.
        """,
        metadata={"source": "deep_learning.txt", "topic": "Deep Learning"}
    ),

    Document(
        page_content="""
        Retrieval-Augmented Generation combines information retrieval
        with a Large Language Model. Relevant documents are retrieved
        from a knowledge base and supplied to the model as context.
        """,
        metadata={"source": "rag.txt", "topic": "RAG"}
    )
]

# loader = DirectoryLoader(
#     "mmr_test",
#     glob="*.txt",
#     loader_cls=TextLoader
# )

# documents = loader.load()

# print(f"Total documents: {len(documents)}")


# --------------------------------------------------
# 2. Create embeddings
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 3. Create vector database
# --------------------------------------------------

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="mmr_test"
)


# --------------------------------------------------
# 4. Query
# --------------------------------------------------

query = "What is Python and why is it popular?"


# ==================================================
# NORMAL SIMILARITY SEARCH
# ==================================================

print("\n" + "=" * 60)
print("NORMAL SIMILARITY SEARCH")
print("=" * 60)

similarity_results = vectorstore.similarity_search(
    query,
    k=4
)

for i, doc in enumerate(similarity_results, 1):

    print(f"\nResult {i}")
    print("-" * 40)

    print("Source:", doc.metadata["source"])
    print("Content:", doc.page_content)


# ==================================================
# MMR SEARCH
# ==================================================

print("\n" + "=" * 60)
print("MMR SEARCH")
print("=" * 60)

# mmr_results = vectorstore.max_marginal_relevance_search(
#     query,
#     k=4,
#     fetch_k=10,
#     lambda_mult=0.5
# )
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 2
    }
)
res=retriever.invoke(query)
for i, doc in enumerate(res, 1):

    print(f"\nResult {i}")
    print("-" * 40)

    print("Source:", doc.metadata["source"])
    print("Content:", doc.page_content)