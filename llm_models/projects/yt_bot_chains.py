from dotenv import load_dotenv
import os

from youtube_transcript_api import YouTubeTranscriptApi

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser


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
# 3. FETCH TRANSCRIPT
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


# ============================================================
# 4. SPLIT TRANSCRIPT INTO DOCUMENTS
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

documents = text_splitter.create_documents(
    [transcript_text]
)

print("Number of documents:", len(documents))


# ============================================================
# 5. CREATE EMBEDDING MODEL
# ============================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 6. CREATE FAISS VECTOR STORE
# ============================================================

vector_store = FAISS.from_documents(
    documents=documents,
    embedding=embedding_model
)


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
# 8. FUNCTION TO FORMAT DOCUMENTS
# ============================================================

def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# ============================================================
# 9. CREATE PROMPT
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
# 10. CREATE LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=groq_api_key,
    temperature=0
)


# ============================================================
# 11. CREATE PARALLEL CHAIN
# ============================================================

parallel_chain = RunnableParallel({

    "question": RunnablePassthrough(),

    "context": (
        retriever
        | RunnableLambda(format_docs)
    )

})


# ============================================================
# 12. CREATE MAIN CHAIN
# ============================================================

main_chain = (
    parallel_chain
    | prompt
    | llm
    | StrOutputParser()
)


# ============================================================
# 13. USER QUESTION
# ============================================================

question = input(
    "\nAsk a question about the YouTube video: "
)


# ============================================================
# 14. INVOKE COMPLETE CHAIN
# ============================================================

response = main_chain.invoke(question)


# ============================================================
# 15. FINAL RESPONSE
# ============================================================

print("\n==============================================")
print("                 AI RESPONSE")
print("==============================================\n")

print(response)