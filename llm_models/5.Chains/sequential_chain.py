from dotenv import load_dotenv
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint
)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    provider="featherless-ai",
    task="text-generation",
    max_new_tokens=150
)

model = ChatHuggingFace(llm=llm)
parser=StrOutputParser()
prompt1=PromptTemplate(
    template="Generate detailed report about topic \n {topic}",
    input_variables=['topic']
)
prompt2=PromptTemplate(
    template="Generate 5 bullet points {text}",
    input_variables=['topic']
)
chain= prompt1 | model | parser | prompt2 | model | parser
res=chain.invoke({'topic': 'employment in india'})
print(res)

chain.get_graph().print_ascii();
