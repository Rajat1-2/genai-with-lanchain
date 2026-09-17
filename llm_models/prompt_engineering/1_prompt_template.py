from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# LLM
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    provider="featherless-ai",
    task="text-generation",
    max_new_tokens=100
)

model = ChatHuggingFace(llm=llm)


# Dynamic Prompt
prompt = PromptTemplate(
    template="""
You are an expert teacher.

Explain the following topic:
Topic: {topic}

Explain it at a {level} level.
Use {language} language.

Keep the explanation clear and easy to understand.
""",
    input_variables=["topic", "level", "language"]
)


# Create prompt with dynamic values
final_prompt = prompt.invoke({
    "topic": "RAG",
    "level": "beginner",
    "language": "simple English"
})


# Send prompt to model
res = model.invoke(final_prompt)

print(res.content)
