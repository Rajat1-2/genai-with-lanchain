from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
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


# Chat history
history = []

while True:

    user_input = input("You : ")

    if user_input.lower() == "exit":
        break

    history.append({
        "role": "user",
        "content": user_input
    })

    res = model.invoke(history)

    history.append({
        "role": "assistant",
        "content": res.content
    })

    print("AI :", res.content)

print(history)