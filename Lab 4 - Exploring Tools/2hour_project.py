"""
LangGraph ReAct Agent with Persistent Multi-Turn Conversation

This program demonstrates a LangGraph application using create_react_agent with:
- A single persistent conversation across multiple turns
- Graph-based looping (no Python loops or checkpointing)
- Automatic conversation history management (trimming after 100 messages)
- Verbose debugging output
"""

import asyncio
import time
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.tools import DuckDuckGoSearchRun


# ============================================================================
# STATE DEFINITION
# ============================================================================

class ConversationState(TypedDict):
    """
    State schema for the conversation.

    Attributes:
        messages: Full conversation history with automatic message merging
        verbose: Controls detailed tracing output
        command: Special command from user (exit, verbose, quiet, or None)
        ddg_result: Raw search results from DuckDuckGo
        wiki_result: Raw article content from Wikipedia
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    verbose: bool
    command: str  # "exit", "verbose", "quiet", or None
    ddg_result: str
    wiki_result: str


# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

@tool
def get_weather(location: str) -> str:
    """
    Get current weather information for a specified location.
    
    Args:
        location: City name or location string
        
    Returns:
        Weather description string
    """
    # Simulate API call delay
    time.sleep(0.5)
    return f"Weather in {location}: Sunny, 72Â°F with light winds"


@tool
def get_population(city: str) -> str:
    """
    Get population information for a specified city.
    
    Args:
        city: City name
        
    Returns:
        Population information string
    """
    # Simulate API call delay
    time.sleep(0.5)
    return f"Population of {city}: Approximately 1 million people"


@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2")
        
    Returns:
        Result of the calculation
    """
    try:
        # Safe evaluation of simple math expressions
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"
    
    
@tool
def wikipedia_tool(query: str) -> str:
    """
    Search Wikipedia for a given query and return a summary.
    
    Args:
        query: Wiki pedia search query
    Returns:
        Summary of the Wikipedia article
    """
    try:
        retriever = WikipediaRetriever()
        docs = retriever.invoke(query)
        return docs[0].page_content if docs else "No results found."
    except Exception as e:
        return f"Error accessing Wikipedia: {str(e)}"
    
@tool
def duckduckgo_search(query: str) -> str:
    """
    Perform a DuckDuckGo search for the given query.
    
    Args:
        query: Search query string
    Returns:
        Search results as a string
    """
    try:
        search_tool = DuckDuckGoSearchRun()
        result = search_tool.run(query)
        return result
    except Exception as e:
        return f"Error performing DuckDuckGo search: {str(e)}"



# List of all available tools
tools = [wikipedia_tool, duckduckgo_search]


# ============================================================================
# NODE FUNCTIONS
# ============================================================================

def input_node(state: ConversationState) -> ConversationState:
    """
    Get input from the user and add it to the conversation.
    
    This node:
    - Prompts the user for input
    - Handles special commands (quit, exit, verbose, quiet)
    - Adds user message to conversation history (for real messages only)
    - Sets command field for special commands
    
    Args:
        state: Current conversation state
        
    Returns:
        Updated state with new user message or command
    """
    if state.get("verbose", True):
        print("\n" + "="*80)
        print("NODE: input_node")
        print("="*80)
    
    # Get user input
    user_input = input("\nYou: ").strip()
    
    # Handle exit commands
    if user_input.lower() in ["quit", "exit"]:
        if state.get("verbose", True):
            print("[DEBUG] Exit command received")
        # Set command field, don't add to messages
        return {"command": "exit"}
    
    # Handle verbose toggle
    if user_input.lower() == "verbose":
        print("[SYSTEM] Verbose mode enabled")
        # Set command field and update verbose flag
        return {"command": "verbose", "verbose": True}
    
    if user_input.lower() == "quiet":
        print("[SYSTEM] Verbose mode disabled")
        # Set command field and update verbose flag
        return {"command": "quiet", "verbose": False}
    
    # Add user message to conversation history
    if state.get("verbose", True):
        print(f"[DEBUG] User input: {user_input}")
    
    # Clear command field and add message
    return {"command": None, "messages": [HumanMessage(content=user_input)]}


def ddg_search_node(state: ConversationState) -> ConversationState:
    """
    Search DuckDuckGo with the user's latest question.

    Extracts the last HumanMessage and runs it through DuckDuckGo.

    Args:
        state: Current conversation state

    Returns:
        Updated state with ddg_result populated
    """
    if state.get("verbose", True):
        print("\n" + "="*80)
        print("NODE: ddg_search_node")
        print("="*80)

    # Get the latest user question
    query = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            query = msg.content
            break

    if state.get("verbose", True):
        print(f"[DEBUG] DuckDuckGo query: {query}")

    try:
        search = DuckDuckGoSearchRun()
        result = search.run(query)
    except Exception as e:
        result = f"Error performing DuckDuckGo search: {str(e)}"

    if state.get("verbose", True):
        print(f"[DEBUG] DDG result preview: {result[:200]}...")

    return {"ddg_result": result}


def wiki_search_node(state: ConversationState) -> ConversationState:
    """
    Search Wikipedia with the user's latest question.

    Extracts the last HumanMessage and fetches the top Wikipedia article.

    Args:
        state: Current conversation state

    Returns:
        Updated state with wiki_result populated
    """
    if state.get("verbose", True):
        print("\n" + "="*80)
        print("NODE: wiki_search_node")
        print("="*80)

    # Get the latest user question
    query = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            query = msg.content
            break

    if state.get("verbose", True):
        print(f"[DEBUG] Wikipedia query: {query}")

    try:
        retriever = WikipediaRetriever()
        docs = retriever.invoke(query)
        result = docs[0].page_content if docs else "No Wikipedia results found."
    except Exception as e:
        result = f"Error accessing Wikipedia: {str(e)}"

    if state.get("verbose", True):
        print(f"[DEBUG] Wikipedia result preview: {result[:200]}...")

    return {"wiki_result": result}


def compare_and_respond(state: ConversationState) -> ConversationState:
    """
    Compare DuckDuckGo and Wikipedia results, then generate a unified LLM response.

    Sends both source results to the LLM with a comparison prompt and adds
    the AI response to the conversation messages.

    Args:
        state: Current conversation state

    Returns:
        Updated state with the AI comparison message appended
    """
    if state.get("verbose", True):
        print("\n" + "="*80)
        print("NODE: compare_and_respond")
        print("="*80)

    global llm

    ddg = state.get("ddg_result", "No DuckDuckGo results.")
    wiki = state.get("wiki_result", "No Wikipedia results.")

    # Get the original question
    question = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            question = msg.content
            break

    comparison_prompt = (
        f"The user asked: \"{question}\"\n\n"
        f"--- DuckDuckGo Search Results ---\n{ddg}\n\n"
        f"--- Wikipedia Results ---\n{wiki}\n\n"
        "Using the two sources above, provide a clear answer to the user's question. "
        "Compare and contrast the information from both sources. "
        "Note any agreements, disagreements, or gaps between them. "
        "Cite which source each piece of information comes from."
    )

    if state.get("verbose", True):
        print(f"[DEBUG] Sending comparison prompt to LLM ({len(comparison_prompt)} chars)")

    response = llm.invoke([SystemMessage(content="You are a helpful research assistant that synthesizes information from multiple sources."),
                           HumanMessage(content=comparison_prompt)])

    if state.get("verbose", True):
        print(f"[DEBUG] LLM response preview: {response.content[:200]}...")

    return {"messages": [response]}


def output_node(state: ConversationState) -> ConversationState:
    """
    Display the assistant's final response to the user.
    
    This node:
    - Extracts the last AI message from the conversation
    - Prints it to the console
    - Returns empty dict (no state changes)
    
    Args:
        state: Current conversation state
        
    Returns:
        Empty dict (no state modifications)
    """
    if state.get("verbose", True):
        print("\n" + "="*80)
        print("NODE: output_node")
        print("="*80)
    
    # Find the last AI message in the conversation
    # (there may be tool messages mixed in)
    last_ai_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            last_ai_message = msg
            break
    
    if last_ai_message:
        print(f"\nAssistant: {last_ai_message.content}")
    else:
        print("\n[WARNING] No assistant response found")
    
    return {}


def trim_history(state: ConversationState) -> ConversationState:
    """
    Manage conversation history length to prevent unlimited growth.
    
    Strategy:
    - Keep the system message (if present)
    - Keep the most recent 100 messages
    - This allows ~50 conversation turns (user + assistant pairs)
    
    Args:
        state: Current conversation state
        
    Returns:
        Updated state with trimmed message history (if needed)
    """
    messages = state["messages"]
    max_messages = 100
    
    # Only trim if we've exceeded the limit
    if len(messages) > max_messages:
        if state.get("verbose", True):
            print(f"\n[DEBUG] History length: {len(messages)} messages")
            print(f"[DEBUG] Trimming to most recent {max_messages} messages")
        
        # Preserve system message if it exists at the start
        if messages and isinstance(messages[0], SystemMessage):
            # Keep system message + last (max_messages - 1) messages
            trimmed = [messages[0]] + list(messages[-(max_messages - 1):])
            if state.get("verbose", True):
                print(f"[DEBUG] Preserved system message + {max_messages - 1} recent messages")
        else:
            # Just keep the last max_messages
            trimmed = list(messages[-max_messages:])
            if state.get("verbose", True):
                print(f"[DEBUG] Kept {max_messages} most recent messages")
        
        return {"messages": trimmed}
    
    # No trimming needed
    return {}


# ============================================================================
# ROUTING LOGIC
# ============================================================================

def route_after_input(state: ConversationState) -> Literal["ddg_search", "end", "input"]:
    """
    Determine where to route after input based on command field.

    Logic:
    - If command is "exit", route to END
    - If command is "verbose" or "quiet", route back to input
    - Otherwise (command is None), route to the DDG search node

    Args:
        state: Current conversation state

    Returns:
        "end" to terminate, "input" for verbose toggle, "ddg_search" to continue pipeline
    """
    command = state.get("command")

    # Check for exit command
    if command == "exit":
        if state.get("verbose", True):
            print("[DEBUG] Routing to END (exit requested)")
        return "end"

    # Check for verbose toggle commands - route back to input
    if command in ["verbose", "quiet"]:
        if state.get("verbose", True):
            print("[DEBUG] Routing back to input (verbose toggle)")
        return "input"

    # Normal message - route to DDG search (start of pipeline)
    if state.get("verbose", True):
        print("[DEBUG] Routing to ddg_search (pipeline start)")
    return "ddg_search"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

# Global variable to hold the LLM (used by compare_and_respond)
llm = None

def create_conversation_graph():
    """
    Build the conversation graph with a deterministic dual-source pipeline.

    Pipeline per question:
        input -> ddg_search -> wiki_search -> compare -> output -> trim -> input (loop)

    Returns:
        Compiled LangGraph application
    """
    global llm

    import os
    import dotenv
    dotenv.load_dotenv()

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    print("[SYSTEM] LLM initialized successfully")

    # ========================================================================
    # Build the Graph
    # ========================================================================

    workflow = StateGraph(ConversationState)

    # Add all nodes
    workflow.add_node("input", input_node)
    workflow.add_node("ddg_search", ddg_search_node)
    workflow.add_node("wiki_search", wiki_search_node)
    workflow.add_node("compare", compare_and_respond)
    workflow.add_node("output", output_node)
    workflow.add_node("trim_history", trim_history)

    # Set entry point
    workflow.set_entry_point("input")

    # Conditional edge from input
    workflow.add_conditional_edges(
        "input",
        route_after_input,
        {
            "ddg_search": "ddg_search",
            "input": "input",       # Loop back for verbose/quiet
            "end": END
        }
    )

    # Linear pipeline: DDG -> Wikipedia -> Compare -> Output -> Trim -> Input
    workflow.add_edge("ddg_search", "wiki_search")
    workflow.add_edge("wiki_search", "compare")
    workflow.add_edge("compare", "output")
    workflow.add_edge("output", "trim_history")
    workflow.add_edge("trim_history", "input")  # Loop!

    return workflow.compile()


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_graph(app):
    """
    Generate a Mermaid diagram for the pipeline graph.

    Creates:
    - langchain_conversation_graph_2_hour_project.png: The dual-source pipeline

    Args:
        app: Compiled conversation graph
    """
    try:
        png = app.get_graph().draw_mermaid_png()
        with open("langchain_conversation_graph_2_hour_project.png", "wb") as f:
            f.write(png)
        print("[SYSTEM] Pipeline graph saved to 'langchain_conversation_graph_2_hour_project.png'")
    except Exception as e:
        print(f"[WARNING] Could not generate graph visualization: {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """
    Main execution function.
    
    This function:
    1. Creates the conversation graph
    2. Visualizes the graph structure
    3. Initializes the conversation state
    4. Invokes the graph ONCE
    
    The graph then runs indefinitely via internal looping (trim_history -> input)
    until the user types 'quit' or 'exit'.
    """
    print("="*80)
    print("LangGraph Dual-Source Research Pipeline")
    print("="*80)
    print("\nPipeline per question:")
    print("  Input -> DuckDuckGo -> Wikipedia -> Compare & Contrast -> Response")
    print("\n  - Every question is searched on BOTH DuckDuckGo and Wikipedia")
    print("  - Results are compared and synthesized by the LLM")
    print("  - History managed automatically (trimmed after 100 messages)")
    print("\nCommands:")
    print("  - Type 'quit' or 'exit' to end the conversation")
    print("  - Type 'verbose' to enable detailed tracing")
    print("  - Type 'quiet' to disable detailed tracing")
    print("="*80)

    # Create the conversation graph
    app = create_conversation_graph()

    # Visualize the pipeline graph
    visualize_graph(app)

    # Initialize conversation state
    initial_state = {
        "messages": [],
        "verbose": True,
        "command": None,
        "ddg_result": "",
        "wiki_result": ""
    }

    print("\n[SYSTEM] Starting conversation...\n")

    try:
        # Invoke the graph ONCE - it loops internally via graph edges
        # Each iteration: input -> ddg_search -> wiki_search -> compare -> output -> trim -> input
        await app.ainvoke(initial_state)
        
    except KeyboardInterrupt:
        print("\n\n[SYSTEM] Interrupted by user (Ctrl+C)")
    
    print("\n[SYSTEM] Conversation ended. Goodbye!\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    asyncio.run(main())