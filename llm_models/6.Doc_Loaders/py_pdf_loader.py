from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("6.Doc_Loaders/rag_sample.pdf")

# Load PDF
docs = loader.load()
print(len(docs))
print(docs[0].page_content)
