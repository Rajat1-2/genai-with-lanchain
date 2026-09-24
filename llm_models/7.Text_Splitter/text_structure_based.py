from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Artificial Intelligence is a field of computer science.

Machine Learning is a subset of Artificial Intelligence.
It allows computers to learn patterns from data.

Deep Learning is a subset of Machine Learning.
It uses neural networks with multiple layers.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk)