from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    provider="featherless-ai",
    task="text-generation",
    max_new_tokens=100
)

model = ChatHuggingFace(llm=llm)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are my AI mentor. Explain concepts clearly "
        "with simple examples."
    ),

    # Placeholder for conversation history
    #  it loads hostory before doing the next stuff , 
    # say if user has started a refuund 
    # it can use history to tell user about the refund process
    #  multi-msg dynamic prompting
    MessagesPlaceholder(variable_name="history"),

    # Current user question
    ("human", "{question}")
])

history = []

while True:

    user_input = input("You : ")

    if user_input.lower() == "exit":
        break

    # Create prompt with history + current question
    messages = prompt.invoke({
        "history": history,
        "question": user_input
    })

    # Send to model
    res = model.invoke(messages)

    print("AI :", res.content)

    # Save conversation
    history.append(
        HumanMessage(content=user_input)
    )

    history.append(
        AIMessage(content=res.content)
    )