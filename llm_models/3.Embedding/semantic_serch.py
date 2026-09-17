from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# 1. Create embedding model
# -----------------------------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# 2. Our documents
# -----------------------------
documents = [
    "Delhi is the capital of India.",
    "Mumbai is the financial capital of India.",
    "The Taj Mahal is located in Agra.",
    "Python is a popular programming language.",
    "Machine learning is a branch of artificial intelligence.",
    "The Himalayas are a mountain range in Asia.",
    "Virat Kohli is a famous Indian cricketer."
]


# -----------------------------
# 3. Create embeddings
# -----------------------------
doc_embeddings = embedding.embed_documents(documents)


# -----------------------------
# 4. Take user query
# -----------------------------
query = "What is the capital city of India?"


# -----------------------------
# 5. Create query embedding
# -----------------------------
query_embedding = embedding.embed_query(query)


# -----------------------------
# 6. Calculate cosine similarity
# -----------------------------
scores = cosine_similarity(
    [query_embedding],
    doc_embeddings
)[0]


# -----------------------------
# 7. Find highest score
# -----------------------------
index, score = sorted(
    list(enumerate(scores)),
    key=lambda x: x[1]
)[-1]


# -----------------------------
# 8. Display result
# -----------------------------
print("Query:", query)
print("Most similar document:", documents[index])
print("Similarity score:", score)