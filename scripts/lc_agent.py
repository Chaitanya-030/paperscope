import truststore
truststore.inject_into_ssl()

import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent

sys.path.append(str(Path(__file__).parent))
from arxiv_tool import search_arxiv as _search_arxiv  # noqa: E402

load_dotenv()

@tool
def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    """Search arXiv for academic papers matching a query.

    Use this when the user asks about research papers, recent work,
    or what's been published on a topic.

    Args:
        query: Search keywords describing the topic of interest.
        max_results: Maximum number of papers to return (default 5, max 20).
    """
    import json
    papers = _search_arxiv(query=query, max_results=max_results)
    if not papers:
        return f"No papers found on arXiv matching query: {query!r}."
    return json.dumps(papers, ensure_ascii=False)

LLM_MODEL = "groq:openai/gpt-oss-20b"
llm = init_chat_model(LLM_MODEL)
agent = create_agent(
        llm,
        tools=[search_arxiv],
        system_prompt=("You are a research assistant. When the user asks about academic papers, "
                        "recent research, or what's been published on a topic, call the `search_arxiv` "
                        "tool using the structured function-call format. Do not write tool invocations "
                        "as plain text. After receiving tool results, summarize them with citations."
                      ),)

if __name__ == "__main__":
    question = "Find me recent papers authored by Chaitanya Sheth."

    result = agent.invoke({"messages": [HumanMessage(question)]})

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(result["messages"][-1].content)

    print("\n" + "=" * 70)
    print(f"MESSAGE TRACE  ({len(result['messages'])} messages)")
    print("=" * 70)
    for i, msg in enumerate(result["messages"], start=1):
        msg_type = type(msg).__name__
        preview = (msg.content[:80] + "...") if msg.content and len(msg.content) > 80 else msg.content
        print(f"  {i}. [{msg_type:15}] {preview!r}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"      tool_call: {tc['name']}({tc['args']})")