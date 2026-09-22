from typing import TypedDict, Annotated
from typing import TypedDict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


class Movie(TypedDict):
    title: Annotated[str, "Name of the movie"]
    director: Annotated[str, "Director of the movie"]
    release_year: Annotated[int, "Year in which the movie was released"]


structured_model = model.with_structured_output(Movie)

response = structured_model.invoke(
    "Tell me about Inception."
)

print(response)