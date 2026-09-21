from dotenv import load_dotenv

from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint
)

from langchain_core.prompts import PromptTemplate
# from langchain.output_parsers import (
#     StructuredOutputParser,
#     ResponseSchema
# )
from langchain_classic.output_parsers.structured import (
    StructuredOutputParser, ResponseSchema
)


load_dotenv()


# -----------------------------
# MODEL
# -----------------------------

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    provider="featherless-ai",
    task="text-generation",
    max_new_tokens=150
)

model = ChatHuggingFace(llm=llm)


# -----------------------------
# RESPONSE SCHEMA
# -----------------------------

response_schemas = [

    ResponseSchema(
        name="name",
        description="Name of the person"
    ),

    ResponseSchema(
        name="occupation",
        description="Occupation or profession"
    ),

    ResponseSchema(
        name="country",
        description="Country associated with the person"
    ),

    ResponseSchema(
        name="summary",
        description="Short summary about the person"
    )
]


# -----------------------------
# STRUCTURED OUTPUT PARSER
# -----------------------------

parser = StructuredOutputParser.from_response_schemas(
    response_schemas
)


# -----------------------------
# FORMAT INSTRUCTIONS
# -----------------------------

format_instructions = parser.get_format_instructions()

print(format_instructions)


# -----------------------------
# PROMPT
# -----------------------------

prompt = PromptTemplate(
    template="""
Give information about the person {person}.

{format_instructions}
""",

    input_variables=["person"],

    partial_variables={
        "format_instructions": format_instructions
    }
)


# -----------------------------
# MANUAL APPROACH
# -----------------------------

# p1 = prompt.invoke({
#     "person": "Donald Trump"
# })
# res = model.invoke(p1)
# print(res.content)
# result = parser.parse(res.content)
# print("\nPARSED RESPONSE:")
# print(result)


# chains
chain=prompt | model | parser
res=chain.invoke({'person':'jeff bejos'})
print(res)