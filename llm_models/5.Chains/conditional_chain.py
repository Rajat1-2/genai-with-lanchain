# from dotenv import load_dotenv
# from typing import Literal

# from pydantic import BaseModel, Field

# from langchain_huggingface import (
#     ChatHuggingFace,
#     HuggingFaceEndpoint
# )

# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import (
#     PydanticOutputParser,
#     StrOutputParser
# )

# from langchain_core.runnables import (
#     RunnableBranch,
#     RunnableLambda,
#     RunnablePassthrough,
# )


# load_dotenv()


# # ============================================================
# # MODEL
# # ============================================================

# llm = HuggingFaceEndpoint(
#     repo_id="Qwen/Qwen2.5-7B-Instruct",
#     provider="featherless-ai",
#     task="text-generation",
#     max_new_tokens=150
# )

# model = ChatHuggingFace(llm=llm)

# str_parser = StrOutputParser()


# # ============================================================
# # 1. PYDANTIC MODEL FOR SENTIMENT
# # ============================================================

# class Sentiment(BaseModel):

#     sentiment: Literal["positive", "negative"] = Field(
#         description="Sentiment of the feedback. "
#                     "It must be either positive or negative."
#     )


# # ============================================================
# # 2. PYDANTIC OUTPUT PARSER
# # ============================================================

# sentiment_parser = PydanticOutputParser(
#     pydantic_object=Sentiment
# )


# # ============================================================
# # 3. SENTIMENT PROMPT
# # ============================================================

# sentiment_prompt = PromptTemplate(
#     template="""
# Analyze the sentiment of the following feedback.

# Feedback:
# {feedback}

# {format_instructions}

# Return only the sentiment as specified.
# """,

#     input_variables=["feedback"],

#     partial_variables={
#         "format_instructions":
#             sentiment_parser.get_format_instructions()
#     }
# )


# # ============================================================
# # 4. SENTIMENT CHAIN
# # ============================================================

# sentiment_chain = (
#     sentiment_prompt
#     | model
#     | sentiment_parser
# )


# # ============================================================
# # 5. POSITIVE RESPONSE CHAIN
# # ============================================================

# positive_prompt = PromptTemplate(
#     template="""
# The user gave the following positive feedback:

# {feedback}

# Write a short, friendly and appreciative response.
# Thank the user for their positive feedback.
# """,

#     input_variables=["feedback"]
# )

# positive_chain = (
#     positive_prompt
#     | model
#     | str_parser
# )


# # ============================================================
# # 6. NEGATIVE RESPONSE CHAIN
# # ============================================================

# negative_prompt = PromptTemplate(
#     template="""
# The user gave the following negative feedback:

# {feedback}

# Write a short, polite and empathetic response.
# Acknowledge the user's concern and apologize if appropriate.
# Do not argue with the user.
# """,

#     input_variables=["feedback"]
# )

# negative_chain = (
#     negative_prompt
#     | model
#     | str_parser
# )


# # ============================================================
# # 7. KEEP ORIGINAL FEEDBACK + ADD SENTIMENT
# # ============================================================

# classification_chain = RunnablePassthrough.assign(
#     sentiment=sentiment_chain
# )


# # ============================================================
# # 8. CONDITIONAL CHAIN
# # ============================================================

# conditional_chain = RunnableBranch(

#     (
#         lambda x: x["sentiment"].sentiment == "positive",
#         positive_prompt | model | str_parser
        
#     ),

#     (
#         lambda x: x["sentiment"].sentiment == "negative",
#         negative_chain
#     ),

#     # Default branch
#     # negative_chain
#     RunnableLambda(lambda x: "could not find sentiment")
# )


# # ============================================================
# # 9. FINAL CHAIN
# # ============================================================

# chain = classification_chain | conditional_chain


# # ============================================================
# # 10. TEST
# # ============================================================

# feedback = input("Enter feedback: ")

# result = chain.invoke({
#     "feedback": feedback
# })

# print("\nFinal Response:")
# print(result)


# # ============================================================
# # 11. GRAPH
# # ============================================================

# chain.get_graph().print_ascii()








from dotenv import load_dotenv
from typing import Literal

from pydantic import BaseModel, Field

from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint
)

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import (
    PydanticOutputParser,
    StrOutputParser
)

from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda
)
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("Set the GROQ_API_KEY environment variable before running this file.") 

# create model
model = ChatGroq(
    model="openai/gpt-oss-20b",  # fast + free model
    api_key=groq_api_key,
    # max_tokens=400,
)

load_dotenv()


# ============================================================
# MODEL
# ============================================================

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    provider="featherless-ai",
    task="text-generation",
    max_new_tokens=150
)

# model = ChatHuggingFace(llm=llm)

str_parser = StrOutputParser()


# ============================================================
# 1. PYDANTIC MODEL FOR SENTIMENT
# ============================================================

class Sentiment(BaseModel):

    sentiment: Literal["positive", "negative"] = Field(
        description=(
            "Sentiment of the feedback. "
            "It must be either positive or negative."
        )
    )


# ============================================================
# 2. PYDANTIC OUTPUT PARSER
# ============================================================

sentiment_parser = PydanticOutputParser(
    pydantic_object=Sentiment
)


# ============================================================
# 3. SENTIMENT PROMPT
# ============================================================

sentiment_prompt = PromptTemplate(
    template="""
Analyze the sentiment of the following feedback.

Feedback:
{feedback}

{format_instructions}

Return only the sentiment as specified.
""",

    input_variables=["feedback"],

    partial_variables={
        "format_instructions":
            sentiment_parser.get_format_instructions()
    }
)


# ============================================================
# 4. SENTIMENT CHAIN
# ============================================================

classification_chain = (
    sentiment_prompt
    | model
    | sentiment_parser
)


# ============================================================
# 5. POSITIVE RESPONSE CHAIN
# ============================================================

positive_prompt = PromptTemplate(
    template="""
The user gave the following positive feedback:

{feedback}

Write a short, friendly and appreciative response.
Thank the user for their positive feedback.
""",

    input_variables=["feedback"]
)

positive_chain = (
    positive_prompt
    | model
    | str_parser
)


# ============================================================
# 6. NEGATIVE RESPONSE CHAIN
# ============================================================

negative_prompt = PromptTemplate(
    template="""
The user gave the following negative feedback:

{feedback}

Write a short, polite and empathetic response.
Acknowledge the user's concern and apologize if appropriate.
Do not argue with the user.
""",

    input_variables=["feedback"]
)

negative_chain = (
    negative_prompt
    | model
    | str_parser
)


# ============================================================
# 7. ADD FEEDBACK + SENTIMENT USING RunnableLambda
# ============================================================

def add_sentiment(data):

    feedback = data["feedback"]

    sentiment = classification_chain.invoke({
        "feedback": feedback
    })

    return {
        "feedback": feedback,
        "sentiment": sentiment
    }


classification_with_feedback = RunnableLambda(
    add_sentiment
)


# ============================================================
# 8. CONDITIONAL CHAIN
# ============================================================

conditional_chain = RunnableBranch(

    (
        lambda x: x["sentiment"].sentiment == "positive",
        positive_chain
    ),

    (
        lambda x: x["sentiment"].sentiment == "negative",
        negative_chain
    ),

    RunnableLambda(
        lambda x: "Could not determine sentiment."
    )
)


# ============================================================
# 9. FINAL CHAIN
# ============================================================

chain = (
    classification_with_feedback
    | conditional_chain
)


# ============================================================
# 10. TEST
# ============================================================

feedback = input("Enter feedback: ")

result = chain.invoke({
    "feedback": feedback
})

print("\nFinal Response:")
print(result)


# ============================================================
# 11. GRAPH
# ============================================================

chain.get_graph().print_ascii()