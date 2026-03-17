#!/usr/bin/env python3
"""
Citation Network Explorer
Builds a structured markdown citation neighborhood report for a seed paper.

Usage:
    python citation_explorer.py ARXIV:2210.03629
    python citation_explorer.py --topic "large language model agents"
"""

import argparse
import json
import os
import sys
from datetime import datetime

import requests
from openai import OpenAI

ASTA_ENDPOINT = "https://asta-tools.allen.ai/mcp/v1"
ASTA_API_KEY = os.environ.get("ASTA_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not ASTA_API_KEY:
    raise ValueError("ASTA_API_KEY environment variable not set")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

client = OpenAI(api_key=OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# MCP plumbing
# ---------------------------------------------------------------------------

def call_mcp(method, params):
    """Send a JSON-RPC 2.0 request to Asta and parse the SSE response."""
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    response = requests.post(
        ASTA_ENDPOINT,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {ASTA_API_KEY}",
        },
        stream=True,
        timeout=120,
    )
    response.raise_for_status()
    for line in response.iter_lines(decode_unicode=True):
        if line.startswith("data:"):
            data = line[5:].strip()
            try:
                return json.loads(data).get("result", {})
            except json.JSONDecodeError:
                continue
    return {}


def extract_result(mcp_result):
    """Pull structured data out of an MCP tools/call response."""
    if not mcp_result:
        return None
    if mcp_result.get("isError"):
        msg = (mcp_result.get("content") or [{}])[0].get("text", "unknown error")
        print(f"  [warning] Tool error: {msg}", file=sys.stderr)
        return None
    structured = mcp_result.get("structuredContent")
    if structured is not None:
        return structured.get("result")
    content = mcp_result.get("content", [])
    texts = [item["text"] for item in content if item.get("type") == "text"]
    if texts:
        try:
            return json.loads(texts[0])
        except json.JSONDecodeError:
            return texts[0]
    return None


def tool_call(name, arguments):
    """Call an Asta tool, log it to stderr, and return the extracted result."""
    # Truncate ids list in log to keep output readable
    log_args = {k: (v[:3] + ["..."] if isinstance(v, list) and len(v) > 3 else v)
                for k, v in arguments.items()}
    print(f"  [mcp] {name}({json.dumps(log_args, ensure_ascii=False)})", file=sys.stderr)
    return extract_result(call_mcp("tools/call", {"name": name, "arguments": arguments}))


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def fetch_seed_paper(seed_id):
    """Step 1: Full metadata for the seed paper, including its reference list."""
    return tool_call("get_paper", {
        "paper_id": seed_id,
        "fields": "paperId,title,abstract,year,authors,fieldsOfStudy,references,citationCount",
    })


def fetch_top_references(seed):
    """Step 2: Fetch abstracts + citation counts for references; return top 5."""
    raw_refs = seed.get("references") or []
    ref_ids = [r["paperId"] for r in raw_refs if r.get("paperId")][:15]
    if not ref_ids:
        return []
    batch = tool_call("get_paper_batch", {
        "ids": ref_ids,
        "fields": "paperId,title,abstract,year,authors,citationCount",
    })
    if not batch or not isinstance(batch, list):
        return []
    valid = [p for p in batch if p and p.get("title")]
    valid.sort(key=lambda p: p.get("citationCount") or 0, reverse=True)
    return valid[:5]


def fetch_recent_citations(seed_id, years_back=3):
    """Step 3: Papers citing the seed paper published in the last N years."""
    cutoff = datetime.now().year - years_back
    result = tool_call("get_citations", {
        "paper_id": seed_id,
        "fields": "paperId,title,year,authors,abstract,citationCount",
        "limit": 5,
        "publication_date_range": f"{cutoff}-01-01:",
    })
    if not result or not isinstance(result, list):
        return []
    papers = []
    for entry in result:
        paper = entry.get("citingPaper", entry)
        if paper.get("title"):
            papers.append(paper)
    return papers[:5]


def fetch_author_profiles(seed, seed_paper_id):
    """Step 4: For each author (up to 3), find their most-cited other work."""
    profiles = []
    for author in (seed.get("authors") or [])[:3]:
        author_id = author.get("authorId")
        if not author_id:
            continue
        papers = tool_call("get_author_papers", {
            "author_id": author_id,
            "paper_fields": "paperId,title,year,citationCount,abstract",
            "limit": 10,
        })
        if not papers or not isinstance(papers, list):
            continue
        others = [p for p in papers if p and p.get("paperId") != seed_paper_id and p.get("title")]
        others.sort(key=lambda p: p.get("citationCount") or 0, reverse=True)
        if others:
            profiles.append({"name": author.get("name"), "top_paper": others[0]})
    return profiles


def detect_recurring_collaborators(top_refs, citing_papers):
    """Bonus: find authors appearing in both references and citing papers."""
    ref_author_ids = {
        a["authorId"]
        for ref in top_refs
        for a in (ref.get("authors") or [])
        if a.get("authorId")
    }
    recurring = []
    for citing in citing_papers:
        for a in (citing.get("authors") or []):
            if a.get("authorId") in ref_author_ids:
                recurring.append({
                    "author": a.get("name"),
                    "citing_paper": citing.get("title"),
                })
    return recurring


def run_pipeline(seed_id):
    print(f"\n[1/4] Fetching seed paper: {seed_id}", file=sys.stderr)
    seed = fetch_seed_paper(seed_id)
    if not seed:
        print(f"Error: could not retrieve paper {seed_id}", file=sys.stderr)
        sys.exit(1)
    print(f"      → {seed.get('title')} ({seed.get('year')})", file=sys.stderr)

    print("[2/4] Fetching top references by citation count...", file=sys.stderr)
    top_refs = fetch_top_references(seed)
    print(f"      → {len(top_refs)} references retrieved", file=sys.stderr)

    print("[3/4] Fetching recent citing papers...", file=sys.stderr)
    citing_papers = fetch_recent_citations(seed_id)
    print(f"      → {len(citing_papers)} citing papers retrieved", file=sys.stderr)

    print("[4/4] Fetching author profiles...", file=sys.stderr)
    seed_paper_id = seed.get("paperId", seed_id)
    author_profiles = fetch_author_profiles(seed, seed_paper_id)
    print(f"      → {len(author_profiles)} author profiles retrieved", file=sys.stderr)

    recurring = detect_recurring_collaborators(top_refs, citing_papers)

    return seed, top_refs, citing_papers, author_profiles, recurring


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(seed, top_refs, citing_papers, author_profiles, recurring):
    """Feed all retrieved data to GPT-4o mini and return a markdown report."""
    print("\n[LLM] Generating markdown report...", file=sys.stderr)

    data = {
        "seed_paper": {
            "title": seed.get("title"),
            "year": seed.get("year"),
            "authors": [a.get("name") for a in (seed.get("authors") or [])],
            "abstract": seed.get("abstract"),
            "fields_of_study": seed.get("fieldsOfStudy") or [],
            "citation_count": seed.get("citationCount"),
        },
        "top_references": [
            {
                "title": r.get("title"),
                "year": r.get("year"),
                "authors": [a.get("name") for a in (r.get("authors") or [])],
                "abstract": r.get("abstract"),
                "citation_count": r.get("citationCount"),
            }
            for r in top_refs
        ],
        "recent_citing_papers": [
            {
                "title": c.get("title"),
                "year": c.get("year"),
                "authors": [a.get("name") for a in (c.get("authors") or [])],
                "abstract": c.get("abstract"),
            }
            for c in citing_papers
        ],
        "author_profiles": [
            {
                "author": p["name"],
                "most_cited_other_work": {
                    "title": p["top_paper"].get("title"),
                    "year": p["top_paper"].get("year"),
                    "citation_count": p["top_paper"].get("citationCount"),
                    "abstract": p["top_paper"].get("abstract"),
                },
            }
            for p in author_profiles
        ],
        "recurring_collaborators": recurring,
    }

    recurring_section = (
        "\n6. **Recurring Collaborations** — authors appearing in both the reference list "
        "and recent citing papers (sign of an active sub-community)"
        if recurring else ""
    )

    prompt = f"""You are a research analyst. Write a structured markdown report about the paper below and its citation network.

Include these sections in order:

1. **Paper Summary** — one paragraph: what the paper proposes, why it matters, its core contribution
2. **Foundational Works** — the 5 most-cited references it builds on (bullet per paper: title, year, 1–2 sentence description of its role)
3. **Recent Developments** — 5 papers from the last 3 years that cite this work (bullet per paper: title, year, brief note on how they extend or apply it)
4. **Author Profiles** — each author's most notable other work (bullet per author: name → title (year, N citations), 1 sentence on what it contributes)
5. **Research Gaps** — 3–5 open problems or underexplored directions you can infer from the gap between the old references and the new citing work{recurring_section}

Citation network data:
```json
{json.dumps(data, indent=2)}
```

Write clear, insightful markdown. Be specific — cite actual titles and authors. Do not pad or repeat yourself."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Bonus: topic → most-cited paper
# ---------------------------------------------------------------------------

def find_top_paper_by_topic(topic):
    """Bonus: search by topic keyword and return the most-cited paper's ID."""
    print(f"Searching for most-cited paper on: '{topic}'", file=sys.stderr)
    papers = tool_call("search_papers_by_relevance", {
        "keyword": topic,
        "fields": "paperId,title,year,citationCount",
        "limit": 10,
    })
    if not papers or not isinstance(papers, list):
        print("Error: no papers found for that topic.", file=sys.stderr)
        sys.exit(1)
    valid = [p for p in papers if p and p.get("paperId")]
    valid.sort(key=lambda p: p.get("citationCount") or 0, reverse=True)
    top = valid[0]
    print(f"Using: {top.get('title')} ({top.get('year')}) — {top.get('citationCount')} citations", file=sys.stderr)
    return top["paperId"]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build a citation network markdown report for a seed paper."
    )
    parser.add_argument(
        "paper_id",
        nargs="?",
        default=None,
        help="ArXiv or Semantic Scholar paper ID (e.g. ARXIV:2210.03629)",
    )
    parser.add_argument(
        "--topic",
        default=None,
        help="Search for the most-cited paper on this topic instead of using a paper ID",
    )
    args = parser.parse_args()

    if args.topic and args.paper_id:
        parser.error("Specify either a paper_id or --topic, not both.")
    if not args.topic and not args.paper_id:
        parser.error("Must provide a paper_id or --topic.")

    seed_id = args.paper_id or find_top_paper_by_topic(args.topic)

    seed, top_refs, citing_papers, author_profiles, recurring = run_pipeline(seed_id)
    report = generate_report(seed, top_refs, citing_papers, author_profiles, recurring)

    print(report)


if __name__ == "__main__":
    main()
