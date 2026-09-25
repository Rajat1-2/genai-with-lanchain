# from langchain_community.retrievers import WikipediaRetriever

# ret=WikipediaRetriever(
#     top_k_results=2,
#     lang="en"
# )
# query="who is president of Russia"
# docs=ret.invoke(query)
# print(docs[0].page_content)

import wikipedia

wikipedia.set_user_agent(
    "RajatLangChainLearning/1.0 (educational project)"
)

from langchain_community.retrievers import WikipediaRetriever


ret = WikipediaRetriever(
    top_k_results=2,
    lang="en"
)

query = "who is president of Russia"

docs = ret.invoke(query)

print("Number of documents:", len(docs))

for i, doc in enumerate(docs):
    print(f"\n========== Result {i + 1} ==========")
    print(doc.page_content)
    print("\nMetadata:")
    print(doc.metadata)