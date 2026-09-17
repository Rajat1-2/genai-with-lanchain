from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv();

model=ChatOpenAI(model="gpt-4",
                 temperature=0.5,
                 max_completion_tokens=10
                 );
res=model.invoke("what is capital of india")
print(res.content);