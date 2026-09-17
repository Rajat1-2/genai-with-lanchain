from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# LLM
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    provider="featherless-ai",
    task="text-generation",
    max_new_tokens=30
)

model = ChatHuggingFace(llm=llm)
history=[]
# chatbot h to we want some memory and also msgs
while True:
    user_input=input('You : ')
    history.append(user_input)
    if user_input=='exit':
        break;
    res=model.invoke(history)
    history.append(res.content)
    print('AI :' , res.content)

