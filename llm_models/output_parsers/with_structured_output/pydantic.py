from typing import TypedDict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

from pydantic import BaseModel, Field


class Movie(BaseModel):
    title: str = Field(description="Name of the movie")
    director: str = Field(description="Director of the movie")
    release_year: int = Field(description="Year the movie was released")
    from pydantic import BaseModel, Field


structured_model = model.with_structured_output(Movie)

response = structured_model.invoke(
    "Tell me about Inception."
)

print(response)
print(type(response))