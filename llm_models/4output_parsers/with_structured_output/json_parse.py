from typing import TypedDict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

schema = {
    "title": "Movie",
    "description": "Information about a movie",
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "Name of the movie"
        },
        "director": {
            "type": "string",
            "description": "Director of the movie"
        },
        "release_year": {
            "type": "integer",
            "description": "Year the movie was released"
        }
    },
    "required": [
        "title",
        "director",
        "release_year"
    ]
}

structured_model = model.with_structured_output(schema)

response = structured_model.invoke(
    "Tell me about Inception."
)

print(response)