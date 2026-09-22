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
prompt=PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=['topic']
)
chain= prompt | model | parser
res=chain.invoke({'topic': 'cricket'})
print(res)

chain.get_graph().print_ascii();
