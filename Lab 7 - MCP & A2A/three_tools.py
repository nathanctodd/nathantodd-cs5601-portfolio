import requests
import json
import os

# Asta MCP endpoint
ASTA_ENDPOINT = "https://asta-tools.allen.ai/mcp/v1"  # MCP JSON-RPC endpoint

# Get API key from environment
API_KEY = os.environ.get("ASTA_API_KEY")
if not API_KEY:
    raise ValueError("ASTA_API_KEY environment variable not set")



def search_papers(query, fields="title,abstract,year,authors", limit=5):
    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "search_papers_by_relevance",
            "arguments": {
                "keyword": query,
                "fields": fields,
                "limit": limit
            }
        }
    }

    response = requests.post(
        ASTA_ENDPOINT,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {API_KEY}"
        },
        stream=True,
        timeout=60
    )
    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
        return None

    result = None
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data:"):
            data = line[5:].strip()
            try:
                rpc_obj = json.loads(data)
                mcp_result = rpc_obj.get("result", {})
                if mcp_result.get("isError"):
                    content = mcp_result.get("content", [{}])
                    print("Tool error:", content[0].get("text", "Unknown error"))
                    return None
                structured = mcp_result.get("structuredContent")
                if structured is not None:
                    result = structured.get("result")
                else:
                    content = mcp_result.get("content", [])
                    if content and content[0].get("type") == "text":
                        result = json.loads(content[0]["text"])
                break
            except json.JSONDecodeError:
                continue

    return result


def search_papers_by_relevance(keyword, fields=None, limit=5):
    arguments = {"keyword": keyword, "limit": limit}
    if fields:
        arguments["fields"] = fields

    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "search_papers_by_relevance",
            "arguments": arguments
        }
    }

    response = requests.post(
        ASTA_ENDPOINT,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {API_KEY}"
        },
        stream=True,
        timeout=60
    )
    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
        return None
    else:
        print("Request successful, parsing response...")

    # Parse SSE stream
    result = None
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue

        # Debug print so we can see what the server is sending
        print("SSE:", line)

        if line.startswith("data:"):
            data = line[5:].strip()
            try:
                rpc_obj = json.loads(data)
                mcp_result = rpc_obj.get("result", {})
                if mcp_result.get("isError"):
                    content = mcp_result.get("content", [{}])
                    print("Tool error:", content[0].get("text", "Unknown error"))
                    return None
                structured = mcp_result.get("structuredContent")
                if structured is not None:
                    result = structured.get("result")
                else:
                    content = mcp_result.get("content", [])
                    if content and content[0].get("type") == "text":
                        result = json.loads(content[0]["text"])
                break
            except json.JSONDecodeError:
                continue

    return result


def get_citations(paper_id, fields=None, limit=5, publication_date_range=None):
    arguments = {"paper_id": paper_id, "limit": limit}
    if fields:
        arguments["fields"] = fields
    if publication_date_range:
        arguments["publication_date_range"] = publication_date_range

    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "get_citations",
            "arguments": arguments
        }
    }

    response = requests.post(
        ASTA_ENDPOINT,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {API_KEY}"
        },
        stream=True,
        timeout=60
    )
    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
        return None

    # Parse SSE stream
    result = None
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue

        # Debug print so we can see what the server is sending
        print("SSE:", line)

        if line.startswith("data:"):
            data = line[5:].strip()
            try:
                rpc_obj = json.loads(data)
                mcp_result = rpc_obj.get("result", {})
                if mcp_result.get("isError"):
                    content = mcp_result.get("content", [{}])
                    print("Tool error:", content[0].get("text", "Unknown error"))
                    return None
                structured = mcp_result.get("structuredContent")
                if structured is not None:
                    result = structured.get("result")
                else:
                    content = mcp_result.get("content", [])
                    if content and content[0].get("type") == "text":
                        result = json.loads(content[0]["text"])
                break
            except json.JSONDecodeError:
                continue

    return result


def search_paper_by_title(title, fields=None, publication_date_range=None, venues=None):
    """
        Description: Search for papers by title.
        Required: title (string)
        Optional: fields (string), publication_date_range (string), venues (string)
    """
    arguments = {"title": title}
    if fields:
        arguments["fields"] = fields
    if publication_date_range:
        arguments["publication_date_range"] = publication_date_range
    if venues:
        arguments["venues"] = venues

    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "search_paper_by_title",
            "arguments": arguments
        }
    }

    response = requests.post(
        ASTA_ENDPOINT,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {API_KEY}"
        },
        stream=True,
        timeout=60
    )
    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
        return None

    result = None
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data:"):
            data = line[5:].strip()
            try:
                rpc_obj = json.loads(data)
                mcp_result = rpc_obj.get("result", {})
                if mcp_result.get("isError"):
                    content = mcp_result.get("content", [{}])
                    print("Tool error:", content[0].get("text", "Unknown error"))
                    return None
                structured = mcp_result.get("structuredContent")
                if structured is not None:
                    result = structured.get("result")
                else:
                    content = mcp_result.get("content", [])
                    if content and content[0].get("type") == "text":
                        result = json.loads(content[0]["text"])
                break
            except json.JSONDecodeError:
                continue

    return result




# --- Drill 1: search_papers — Recent LLM Agent Papers ---
print("=== Drill 1: search_papers_by_relevance — Recent LLM Agent Papers ===")
response = search_papers_by_relevance("large language model agents", fields="title,abstract,year,authors", limit=5)
if response:
    papers = response if isinstance(response, list) else response.get("result", [])
    for i, paper in enumerate(papers, 1):
        print(f"{i}. {paper.get('title')} ({paper.get('year')})")

# --- Drill 2: get_citations — BERT Paper Impact (2023+) ---
print("\n=== Drill 2: get_citations — BERT Paper Impact (2023+) ===")
response = get_citations(
    "ARXIV:1810.04805",
    fields="title,year,authors",
    limit=10,
    publication_date_range="2023-01-01:"
)
if response:
    citations = response if isinstance(response, list) else []
    print(f"Found {len(citations)} citing papers (2023+)")
    for entry in citations[:5]:
        paper = entry.get("citingPaper", entry)
        print(f"  - {paper.get('title')} ({paper.get('year')})")

# --- Drill 3: search_paper_by_title — ReAct Paper (get_references unavailable) ---
# get_references is not available in Asta; using search_paper_by_title as substitute
print("\n=== Drill 3: search_paper_by_title — ReAct Paper ===")
response = search_paper_by_title(
    "ReAct: Synergizing Reasoning and Acting in Language Models",
    fields="title,year,authors"
)
if response:
    papers = response if isinstance(response, list) else [response]
    for i, paper in enumerate(papers, 1):
        authors = [a.get("name", "") for a in paper.get("authors", [])]
        print(f"{i}. {paper.get('title')} ({paper.get('year')})")
        print(f"   Authors: {', '.join(authors)}")
        print("-" * 40)