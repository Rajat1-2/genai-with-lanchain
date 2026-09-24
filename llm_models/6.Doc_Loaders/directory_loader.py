from langchain_community.document_loaders import DirectoryLoader

# Create loader
loader = DirectoryLoader(
    "6.Doc_Loaders/documents",
    # glob="*.txt" # it tell what type of files it has to open
    glob="*",
    # loader_cls=Pypdf and more
)

# Load all documents
docs = loader.load()

print("Number of documents:", len(docs))

for i, doc in enumerate(docs):

    print("\n==============================")
    print("Document:", i + 1)
    print("==============================")

    print("Content:")
    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)