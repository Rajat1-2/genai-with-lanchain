from langchain_text_splitters import CharacterTextSplitter

text = """
Artificial Intelligence is a field of computer science.Machine Learning is a subset of Artificial Intelligence.Deep Learning is a subset of Machine Learning.
Generative AI can generate new content such as text and images.
"""

splitter = CharacterTextSplitter(
    chunk_size=30,
    chunk_overlap=0,
    separator="\n"
)

chunks = splitter.split_text(text)

print(chunks)
print(type(chunks))
print(chunks[0])