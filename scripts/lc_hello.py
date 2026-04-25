import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage


# `init_chat_model` is the provider-agnostic factory.
# "groq:llama-3.3-70b-versatile" tells LangChain to pick the langchain-groq integration
# and use that model.
llm = init_chat_model("groq:llama-3.3-70b-versatile")

if __name__ == "__main__":
    print("[lc] sending 'hello' to Llama via LangChain...")
    response = llm.invoke([HumanMessage("Say hi in 5 words or fewer.")])
    print(f"[lc] response: {response.content!r}")
    print(f"[lc] usage:    {response.usage_metadata}")