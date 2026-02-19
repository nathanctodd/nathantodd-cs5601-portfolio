
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import base64
import os

# ===============================
# Model Setup (LLaVA via Ollama)
# ===============================
# Make sure:
# 1. `ollama serve` is running
# 2. `ollama pull llava` is completed
llm = ChatOllama(model="llava", temperature=0)


# ===============================
# LangGraph State Definition
# ===============================
class ChatState(TypedDict):
    messages: List


def call_model(state: ChatState):
    """LangGraph node that calls the LLM."""
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [AIMessage(content=response.content)]}


# ===============================
# Build LangGraph
# ===============================
graph = StateGraph(ChatState)
graph.add_node("model", call_model)
graph.set_entry_point("model")
graph.add_edge("model", END)

app = graph.compile()


# ===============================
# Image + Conversation Handling
# ===============================
image_b64 = None
image_mime = "image/jpeg"
chat_history = []


def load_image(image_path: str):
    """Load and encode image to base64."""
    global image_b64, image_mime, chat_history

    if not os.path.exists(image_path):
        print(f"Error: File {image_path} not found")
        return

    with open(image_path, "rb") as f:
        raw = f.read()

    ext = os.path.splitext(image_path)[1].lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
    }
    image_mime = mime_types.get(ext, "image/jpeg")

    image_b64 = base64.b64encode(raw).decode("utf-8")
    chat_history = []

    print(f"Loaded image: {os.path.basename(image_path)}")


def ask_question(user_text: str):
    """Send question to LangGraph app."""
    global image_b64, image_mime, chat_history

    if not user_text.strip():
        return

    if image_b64 is None:
        print("Please load an image first using load_image(path)")
        return

    # First message must include image
    if not chat_history:
        human_msg = HumanMessage(
            content=[
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": f"data:{image_mime};base64,{image_b64}",
                },
            ]
        )
    else:
        human_msg = HumanMessage(content=user_text)

    state = {"messages": chat_history + [human_msg]}
    result = app.invoke(state)

    chat_history = result["messages"]

    print(f"\nYou: {user_text}")
    print(f"Assistant: {chat_history[-1].content}")


# ===============================
# CLI Chat Loop
# ===============================
if __name__ == "__main__":
    path = input("Enter image path: ").strip()
    load_image(path)

    print("\nStart chatting about the image (type 'exit' to quit)\n")

    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit"]:
            break
        ask_question(question)
