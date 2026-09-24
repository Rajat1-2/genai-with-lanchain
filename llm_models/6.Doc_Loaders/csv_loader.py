from langchain_community.document_loaders import CSVLoader

# Create loader
loader = CSVLoader("6.Doc_Loaders/students.csv")

# Load CSV
docs = loader.load()

print(docs)