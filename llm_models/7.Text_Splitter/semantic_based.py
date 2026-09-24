from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Semantic splitter
splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=1
)

text = """
Artificial Intelligence is a field of computer science that focuses on creating
machines capable of performing tasks that normally require human intelligence.

Machine learning is a subset of artificial intelligence. It allows computers
to learn patterns from data and make predictions without being explicitly
programmed for every task.

Deep learning is a subset of machine learning that uses neural networks with
multiple layers. It is widely used in computer vision and natural language
processing.

The Eiffel Tower is located in Paris, France. It was completed in 1889 and
has become one of the most recognizable landmarks in the world.
"""

# Split text based on semantic meaning
chunks = splitter.split_text(text)

print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n========== CHUNK {i + 1} ==========")
    print(chunk)