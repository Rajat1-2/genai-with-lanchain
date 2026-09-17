from dotenv import load_dotenv
from langchain_openai import OpenAI
load_dotenv()
llm=OpenAI(model="")
res=llm.invoke("what is capital of india");
print(res.content)