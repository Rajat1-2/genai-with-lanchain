from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("6.Doc_Loaders/dummy.txt")

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)

chunks = splitter.split_documents(docs)

print("Original documents:", len(docs))
print("Chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk.page_content)
    print(chunk.metadata)