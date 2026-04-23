"""
A small end-to-end agent: Llama 3.3 (Groq) calls a `search_arxiv` tool, reads
the results, and composes a final answer. ~100 lines, no agent framework.
"""

# truststore must be injected before any HTTPS-using import (groq, arxiv, ...)
import truststore
truststore.inject_into_ssl()

import json
import logging
import time

import arxiv
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


# ----- Config -----

MODEL = "llama-3.3-70b-versatile"
MAX_ITERATIONS = 5
VERBOSE = False  # set True for arxiv + httpx debug logs

if VERBOSE:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    logging.getLogger("arxiv").setLevel(logging.DEBUG)


# ----- Tool: arXiv search -----

def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    """Search arXiv for papers matching `query`. Returns a list of dicts."""
    print(f"[arxiv] {query!r} (n={max_results})")
    t0 = time.time()

    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=1)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    papers: list[dict] = []
    try:
        for result in client.results(search):
            papers.append({
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "abstract": result.summary.strip().replace("\n", " "),
                "url": result.entry_id,
                "published": result.published.strftime("%Y-%m-%d"),
            })
    except arxiv.HTTPError as e:
        msg = f"[arxiv] HTTP {e.status} after {time.time() - t0:.1f}s"
        if e.status == 429:
            msg += " — rate limited; wait ~10 minutes before retrying"
        print(msg)
        raise

    print(f"[arxiv] done in {time.time() - t0:.1f}s — {len(papers)} papers")
    return papers


# ----- Tool registry: schema for the LLM, mapping for the runtime -----

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_arxiv",
            "description": (
                "Search arXiv for academic papers matching a query. Returns a list of papers "
                "with title, authors, abstract, URL, and publication date. Use this when the "
                "user asks about research papers, recent work, or what's been published on a topic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query — keywords describing the topic of interest.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of papers to return. Default 5, max 20.",
                    },
                },
                "required": ["query"],
            },
        },
    }
]

TOOL_FUNCTIONS = {"search_arxiv": search_arxiv}


# ----- Agent loop -----

def run_agent(question: str) -> str:
    """Loop the LLM with tools until it produces a text answer (or hits the cap)."""
    client = Groq()
    messages = [{"role": "user", "content": question}]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n>>> iteration {iteration}")
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=800,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            print(">>> final answer ready")
            return msg.content or ""

        print(f">>> {len(msg.tool_calls)} tool call(s)")
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })

        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            print(f">>>   {fn_name}({fn_args})")

            fn = TOOL_FUNCTIONS[fn_name]
            try:
                content = json.dumps(fn(**fn_args), ensure_ascii=False)
            except Exception as e:
                content = json.dumps({"error": str(e)})

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": content,
            })

    return "[max iterations reached without final answer]"


if __name__ == "__main__":
    question = "Find me 3 recent papers about UAVs and WSNs and tell me which one looks most novel."
    answer = run_agent(question)
    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(answer)