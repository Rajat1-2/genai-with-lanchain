from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal

from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint
)

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


load_dotenv()


# -----------------------------
# MODEL
# -----------------------------

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    provider="featherless-ai",
    task="text-generation",
    max_new_tokens=200
)

model = ChatHuggingFace(llm=llm)


# -----------------------------
# PYDANTIC MODEL
# -----------------------------

class Person(BaseModel):

    name: str = Field(
        description="Full name of the person"
    )

    age: int = Field(
        description="Age of the person"
    )

    city: str = Field(
        description="City where the person lives"
    )

    gender: Literal["male", "female", "other"] = Field(
        description="Gender of the person"
    )


# -----------------------------
# PARSER
# -----------------------------

parser = PydanticOutputParser(
    pydantic_object=Person
)


# -----------------------------
# PROMPT
# -----------------------------

prompt = PromptTemplate(
    template="""
Give information about a person from {place}.

{format_instructions}

Return accurate information and follow the format exactly.
""",

    input_variables=["place"],

    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)


# -----------------------------
# CHAIN
# -----------------------------

chain = prompt | model | parser


# -----------------------------
# INVOKE
# -----------------------------

res = chain.invoke({
    "place": "India"
})

print(res)
print(type(res))

print("\nName:", res.name)
print("Age:", res.age)
print("City:", res.city)
print("Gender:", res.gender)