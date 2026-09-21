from dotenv import load_dotenv

from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint
)

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    provider="featherless-ai",
    task="text-generation",
    max_new_tokens=100
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()


prompt = PromptTemplate(
    template="""
Give information about the topic {topic},{format_instruction}
""",
    input_variables=["topic"],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

p1=prompt.invoke({'topic':'Donald trump'})
res=model.invoke(p1);
# res have unordered json format
result=parser.parse(res.content)
print(result)





# chain = prompt | model | parser

# result = chain.invoke({
#     "topic": "black hole"
# })

# result = chain.invoke({})   # if no input is there pass the empty dictionary

# print(result)
# print(type(result))