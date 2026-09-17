from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
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
history=[
    SystemMessage(content="hi you are my ai mentor"),
    # HumanMessage(content="tell me about VAE encoders")
]
# chatbot h to we want some memory and also msgs
while True:
    user_input=input('You : ')
    history.append(HumanMessage(content=user_input))
    if user_input=='exit':
        break;
    res=model.invoke(history)
    history.append(AIMessage(content=res.content))
    print('AI :' , res.content)

print(history)