"""
A2A Agent Starter Template
===========================
This template sets up everything you need to run an A2A-compatible agent:
  - A FastAPI web server with an Agent Card endpoint
  - Automatic ngrok URL detection
  - Automatic registration with the class registry
  - A /task endpoint where your agent receives questions and responds

YOUR JOB:
  1. Edit the AGENT_CONFIG section below (name, description, skills)
  2. Edit the handle_task() function to implement your agent's logic
  3. Start ngrok in a separate terminal:  ngrok http 8000
  4. Run this script:  python a2a_agent_template.py

DRY RUN MODE (for testing your system prompt without ngrok or the registry):
  python a2a_agent_template.py --dryrun

DEPENDENCIES:
  pip install fastapi uvicorn requests openai python-dotenv
"""

import os
import json
import requests
from fastapi import FastAPI, Request
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# =============================================================================
# âœï¸  EDIT THIS SECTION â€” Define your agent's identity and skills
# =============================================================================

AGENT_CONFIG = {
    "name": "Nathan and Anna's Movies and TV Agent",          # e.g., "Alice's History Agent"
    "description": "An agent that answers questions about movies and TV shows, including plot summaries, cast information, and recommendations.",  # A brief description of what your agent does
    "skills": [
        {
            "id": "movie-qa-skill",               # a short unique id
            "name": "Movie Q&A",          # e.g., "History Q&A"
            "description": "Answer questions about movies and TV shows.",
        },
        {
            "id": "tv-qa-skill",
            "name": "TV Show Q&A",
            "description": "Answer questions about TV shows.",
        },
        {
            "id": "movie-recommendation-skill",
            "name": "Movie Recommendations",
            "description": "Recommend movies and TV shows based on user preferences.",
        },
        {
            "id": "cast-info-skill",
            "name": "Cast Information",
            "description": "Provide information about the cast of movies and TV shows.",
        },
        {
            "id": "plot-summary-skill",
            "name": "Plot Summaries",
            "description": "Give plot summaries for movies and TV shows.",
        }
    ],
}

# The system prompt tells the LLM how to behave as your agent.
# Customize this to match your agent's specialty.
SYSTEM_PROMPT = """You are Nathan and Anna's Movies and TV Agent, a helpful entertainment assistant with five core tools/capabilities:
1. Movie Q&A: answer questions about movies, including plot details, themes, release info, and general background.
2. TV Show Q&A: answer questions about TV shows, including seasons, characters, plotlines, and episode context.
3. Movie Recommendations: recommend movies or TV shows based on genre, mood, favorite titles, actors, or viewing preferences.
4. Cast Information: provide information about actors, actresses, directors, and the cast of movies and TV shows.
5. Plot Summaries: give concise spoiler-light or spoiler-inclusive plot summaries for movies and TV shows depending on the user's request.

Important behavior rules:
- Be conversational, accurate, and concise.
- When recommending titles, explain why each recommendation fits the user's preferences.
- When discussing plot summaries, avoid spoilers unless the user asks for them.
- If the user asks for cast information, clearly distinguish between actors, characters, and creators when relevant.
- Prefer structured, practical answers.
- When useful, organize the response under these headings:
  - Best answer
  - Recommendations
  - Cast or creator notes
  - Spoiler warning

When answering, infer which capability/tool is most relevant and tailor the response accordingly."""

# =============================================================================
# âš™ï¸  CONFIGURATION â€” You probably don't need to change these
# =============================================================================

REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:8001")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
PORT = int(os.getenv("PORT", "8000"))

def classify_entertainment_tool(question: str) -> str:
    """Very lightweight router so the agent can explicitly use one of its listed tools."""
    q = question.lower()

    recommendation_keywords = [
        "recommend", "recommendation", "similar to", "what should i watch",
        "suggest", "looking for", "best movies", "best shows", "watch next"
    ]
    cast_keywords = [
        "cast", "actor", "actors", "actress", "actresses", "director",
        "creator", "who plays", "who starred", "who is in", "starring"
    ]
    plot_keywords = [
        "plot", "summary", "synopsis", "what happens", "ending", "spoiler",
        "recap", "story"
    ]
    tv_keywords = [
        "tv", "show", "series", "season", "episode", "episodes", "sitcom",
        "miniseries"
    ]
    movie_keywords = [
        "movie", "film", "cinema"
    ]

    if any(k in q for k in recommendation_keywords):
        return "movie-recommendation-skill"
    if any(k in q for k in cast_keywords):
        return "cast-info-skill"
    if any(k in q for k in plot_keywords):
        return "plot-summary-skill"
    if any(k in q for k in tv_keywords):
        return "tv-qa-skill"
    if any(k in q for k in movie_keywords):
        return "movie-qa-skill"
    return "movie-qa-skill"

# =============================================================================
#  INFRASTRUCTURE - No need to edit below this line
# =============================================================================

app = FastAPI()
client = OpenAI(api_key=OPENAI_API_KEY)

# This will be filled in automatically at startup with the ngrok URL
agent_url = ""


# --- Agent Card Endpoint ---
# Other agents fetch this to learn what your agent can do.

@app.get("/.well-known/agent.json")
async def agent_card():
    return {
        "name": AGENT_CONFIG["name"],
        "description": AGENT_CONFIG["description"],
        "url": agent_url,
        "skills": AGENT_CONFIG["skills"],
    }


# --- Task Endpoint ---
# Other agents send tasks here. This is where your agent does its work.

@app.post("/task")
async def receive_task(request: Request):
    body = await request.json()
    question = body.get("question", "")
    sender = body.get("sender", "unknown")

    print(f"\nReceived task from {sender}: {question}")

    answer = handle_task(question)

    print(f"Responding: {answer[:100]}...")

    return {
        "agent": AGENT_CONFIG["name"],
        "answer": answer,
    }


# --- Health Check ---
# The registry can ping this to check if your agent is still alive.

@app.get("/health")
async def health():
    return {"status": "ok", "agent": AGENT_CONFIG["name"]}


# =============================================================================
#  EDIT THIS FUNCTION - This is your agent's brain
# =============================================================================

def handle_task(question: str) -> str:
    """
    This function is called when your agent receives a task.
    It routes the question to one of the agent's listed capabilities and
    includes that tool choice in the system context.
    """
    try:
        tool_id = classify_entertainment_tool(question)
        tool_lookup = {skill["id"]: skill for skill in AGENT_CONFIG["skills"]}
        selected_tool = tool_lookup.get(tool_id, AGENT_CONFIG["skills"][0])

        tool_context = f"""
            Selected capability:
            - Tool ID: {selected_tool['id']}
            - Tool Name: {selected_tool['name']}
            - Tool Description: {selected_tool['description']}

            Use this capability as the primary lens for answering the user's question.
            """

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "system", "content": tool_context},
                {"role": "user", "content": question},
            ],
        )
        return response.choices[0].message.content

    except Exception as e:
        error_msg = f"Error generating response: {e}"
        print(f"❌ {error_msg}")
        return error_msg


# =============================================================================
#  STARTUP - Detects ngrok URL and registers with the class registry
# =============================================================================

def get_ngrok_url() -> str:
    """Read the public URL from ngrok's local API."""
    try:
        resp = requests.get("http://localhost:4040/api/tunnels", timeout=5)
        tunnels = resp.json().get("tunnels", [])
        for tunnel in tunnels:
            if tunnel.get("proto") == "https":
                return tunnel["public_url"]
        if tunnels:
            return tunnels[0]["public_url"]
    except requests.exceptions.ConnectionError:
        print("No ngrok tunnels found.")
        print("   Start ngrok first:  ngrok http 8000")
        raise SystemExit(1)
    except Exception as e:
        print(f"Error reading ngrok URL: {e}")
        raise SystemExit(1)

    print("No ngrok tunnels found.")
    raise SystemExit(1)


def register_with_registry(url: str):
    """Register this agent with the class registry."""
    try:
        resp = requests.post(
            f"{REGISTRY_URL}/register",
            json={
                "name": AGENT_CONFIG["name"],
                "url": url,
                "description": AGENT_CONFIG["description"],
                "skills": AGENT_CONFIG["skills"],
            },
            timeout=5,
        )
        if resp.status_code == 200:
            print(f"Registered with registry at {REGISTRY_URL}")
        else:
            print(f"Registry responded with status {resp.status_code}: {resp.text}")
    except requests.exceptions.ConnectionError:
        print(f"Could not reach registry at {REGISTRY_URL} continuing anyway.")
        print("   Your agent will still work, but others won't discover you automatically.")
    except Exception as e:
        print(f"Registration error: {e} continuing anyway.")


def startup():
    """Detect ngrok URL, register, and print status."""
    global agent_url

    print("=" * 60)
    print(f"Starting: {AGENT_CONFIG['name']}")
    print("=" * 60)

    # Step 1: Get ngrok URL
    agent_url = get_ngrok_url()
    print(f"Public URL: {agent_url}")

    # Step 2: Register with the class registry
    register_with_registry(agent_url)

    # Step 3: Print summary
    print(f"\nAgent Card: {agent_url}/.well-known/agent.json")
    print(f"Task endpoint: {agent_url}/task")
    print(f"Skills: {', '.join(s['name'] for s in AGENT_CONFIG['skills'])}")
    print(f"\nReady to receive tasks!\n")


# =============================================================================
# ðŸ§ª  DRY RUN MODE â€” Test your system prompt locally without ngrok/registry
# =============================================================================

def dryrun():
    """Interactive loop: type questions, see your agent's responses."""
    print("=" * 60)
    print(f"ðŸ§ª DRY RUN: {AGENT_CONFIG['name']}")
    print("=" * 60)
    print(f"   Testing your agent locally â€” no ngrok or registry needed.")
    print(f"   Type a question and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            question = input("ðŸ“ Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        selected_tool = classify_entertainment_tool(question)
        print(f"Thinking... using tool: {selected_tool}")
        answer = handle_task(question)
        print(f"{AGENT_CONFIG['name']}: {answer}\n")


# =============================================================================
# ðŸ  MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="A2A Agent")
    parser.add_argument("--dryrun", action="store_true",
                        help="Test your agent locally â€” type questions, see responses. "
                             "No ngrok or registry needed.")
    args = parser.parse_args()

    if args.dryrun:
        dryrun()
    else:
        startup()
        uvicorn.run(app, host="0.0.0.0", port=PORT)