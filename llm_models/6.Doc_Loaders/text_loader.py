from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("Set the GROQ_API_KEY environment variable before running this file.") 

# create model
llm = ChatGroq(
    model="openai/gpt-oss-20b",  # fast + free model
    api_key=groq_api_key,
)
# ask question
response = llm.invoke("Explain AI in simple words")
print(response.content)