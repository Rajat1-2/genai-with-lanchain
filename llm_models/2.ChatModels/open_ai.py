from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
parser=StrOutputParser()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    provider="featherless-ai",
    task="text-generation",
    max_new_tokens=100
)
model=ChatHuggingFace(llm=llm)
prompt1=PromptTemplate(
    template='detailed report on {topic}',
    input_variables=['topic'],
)
prompt2=PromptTemplate(
    template='Generate a 5 line summary from text \n {text}',
    input_variables=['text'],
)
# p1=prompt1.invoke({'topic':"black hole"});
# r1=model.invoke(p1);

# p2=prompt2.invoke({'text':r1.content});
# r2=model.invoke(p2);
# # print(r2)
# res=parser.parse(r2)
# print(res.content);


# chains
chain= prompt1 | model | parser | prompt2 | model | parser;
res=chain.invoke({'topic': 'black hole'})
print(res)
