from typing import TypedDict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


class Movie(TypedDict):
    title: str
    director: str
    release_year: int


structured_model = model.with_structured_output(Movie)

response = structured_model.invoke(
    "Give me information about the movie Inception."
)

print(response)
print(type(response))