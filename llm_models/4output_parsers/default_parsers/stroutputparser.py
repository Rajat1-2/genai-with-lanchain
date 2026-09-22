
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# load_dotenv()from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

parser = StrOutputParser()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    provider="featherless-ai",
    task="text-generation",
    max_new_tokens=100
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="detailed report on {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Generate a 5 line summary from text:\n{text}",
    input_variables=["text"],
)

# First chain
chain1 = prompt1 | model

r1 = chain1.invoke({
    "topic": "black hole"
})

print("R1:")
print(r1)
print(type(r1))

# Second chain + parser
chain2 = prompt2 | model | parser

r2 = chain2.invoke({
    "text": r1.content
})

print("\nR2:")
print(r2)
print(type(r2))



# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# load_dotenv()
# parser=StrOutputParser()

# llm = HuggingFaceEndpoint(
#     repo_id="Qwen/Qwen2.5-7B-Instruct",
#     provider="featherless-ai",
#     task="text-generation",
#     max_new_tokens=100
# )
# model=ChatHuggingFace(llm=llm)
# prompt1=PromptTemplate(
#     template='detailed report on {topic}',
#     input_variables=['topic'],
# )
# prompt2=PromptTemplate(
#     template='Generate a 5 line summary from text \n {text}',
#     input_variables=['text'],
# )
# p1=prompt1.invoke({'topic':"black hole"})*;*
# r1=model.invoke(p1)*;*

# p2=prompt2.invoke({'text':r1.content})*;*
# r2=model.invoke(p2)*;*
# \# print(r2)
# res=parser.parse(r2)
# print(res)*;but it gives the res with metadata the strparser doent work*












# model = ChatOpenAI(
#     model="gpt-4o-mini",
#     temperature=0
# )

# parser = StrOutputParser()

# response = model.invoke(
#     "What is artificial intelligence?"
# )

# print(response)
# print(type(response))

# result = parser.invoke(response)

# print(result)
# print(type(result))

