# from langchain_community.document_loaders import WebBaseLoader

# loader = WebBaseLoader('https://deepmind.google/discover/blog/alphafold-a-solution-to-a-50-year-old-grand-challenge-in-biology/?utm_source=chatgpt.com')
# docs = loader.load()
# print(docs[0].page_content)
from langchain_community.document_loaders import WebBaseLoader

url = "https://deepmind.google/discover/blog/alphafold-a-solution-to-a-50-year-old-grand-challenge-in-biology/"

loader = WebBaseLoader(url)

docs = loader.load()

print("Number of documents:", len(docs))
print(docs[0].page_content)
print(docs[0].metadata)