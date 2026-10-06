from dotenv import load_dotenv
import os

from youtube_transcript_api import YouTubeTranscriptApi

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set in .env")


# ============================================================
# 2. YOUTUBE VIDEO
# ============================================================

video_id = "J5_-l7WIO_w"


# ============================================================
# 3. FETCH YOUTUBE TRANSCRIPT
# ============================================================

print("Fetching transcript...")

api = YouTubeTranscriptApi()

transcript = api.fetch(
    video_id,
    languages=["en"]
)

transcript_text = " ".join(
    snippet.text
    for snippet in transcript
)

print("Transcript fetched successfully.")
print("Transcript length:", len(transcript_text))


# ============================================================
# 4. SPLIT TRANSCRIPT INTO CHUNKS
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


# create_documents() directly creates Document objects
documents = text_splitter.create_documents(
    [transcript_text]
)

print("Number of documents:", len(documents))


# ============================================================
# 5. CREATE EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ============================================================
# 6. CREATE FAISS VECTOR STORE
# ============================================================

print("Creating FAISS vector store...")

vector_store = FAISS.from_documents(
    documents=documents,
    embedding=embedding_model
)

print("FAISS vector store created.")


# ============================================================
# 7. CREATE RETRIEVER
# ============================================================

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3
    }
)


# ============================================================
# 8. USER QUESTION
# ============================================================

question = input(
    "\nwhy rag is used : "
)


# ============================================================
# 9. RETRIEVE RELEVANT DOCUMENTS
# ============================================================

results = retriever.invoke(question)


print("\n================ RETRIEVED DOCUMENTS ================")

for i, doc in enumerate(results, 1):

    print(f"\n----- Document {i} -----")

    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)


# ============================================================
# 10. CREATE CONTEXT
# ============================================================

context = "\n\n".join(
    doc.page_content
    for doc in results
)


# ============================================================
# 11. CREATE PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Answer the user's question using ONLY the
provided context from the YouTube video.

If the answer cannot be found in the context,
say:

"I don't know based on the provided video."

Context:
{context}

Question:
{question}

Answer:
"""
)


# ============================================================
# 12. CREATE LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=groq_api_key,
    temperature=0
)


# ============================================================
# 13. FORMAT PROMPT
# ============================================================

formatted_prompt = prompt.invoke(
    {
        "context": context,
        "question": question
    }
)


# ============================================================
# 14. CALL LLM
# ============================================================

response = llm.invoke(
    formatted_prompt
)


# ============================================================
# 15. FINAL RESPONSE
# ============================================================

print("\n================================================")
print("                 AI RESPONSE")
print("================================================\n")

print(response.content)