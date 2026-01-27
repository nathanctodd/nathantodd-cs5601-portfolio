"""
Tool Calling with LangChain + LangGraph

Uses LangGraph nodes and edges instead of a Python for-loop for the agent.
Runs a single long conversation with checkpointing and recovery via SqliteSaver.

Graph structure:
    START -> get_user_input -> [route] -> call_llm -> execute_tools -> route_after_tools -> ...
                                  |                                          |
                                  +-> END (quit)            call_llm <-------+  (tool results need another LLM pass)
                                                                             |
                                                            print_response <-+  (no more tool calls)
                                                                  |
                                                            get_user_input (loop)
"""

import os
import sqlite3
from typing import TypedDict, List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
load_dotenv()


# ============================================
# PART 1: Define Your Tools
# ============================================

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location"""
    weather_data = {
        "San Francisco": "Sunny, 72°F",
        "New York": "Cloudy, 55°F",
        "London": "Rainy, 48°F",
        "Tokyo": "Clear, 65°F"
    }
    return weather_data.get(location, f"Weather data not available for {location}")


@tool
def calculate_geometric_function(function: str, angle_type: str, angle: float) -> str:
    """Calculates sin, cos, or tan of a given angle in degrees or radians"""
    import math

    if angle_type == "degrees":
        angle_rad = math.radians(angle)
    elif angle_type == "radians":
        angle_rad = angle
    else:
        return "Error: angle_type must be 'degrees' or 'radians'"

    if function == "sin":
        result = math.sin(angle_rad)
    elif function == "cos":
        result = math.cos(angle_rad)
    elif function == "tan":
        result = math.tan(angle_rad)
    else:
        return "Error: function must be 'sin', 'cos', or 'tan'"

    return f"The {function} of {angle} {angle_type} is {result}"


@tool
def count_character_in_string(input_string: str, character: str) -> str:
    """Counts occurrences of a character in a string"""
    count = input_string.count(character)
    return f"The character '{character}' occurs {count} times in the given string."


@tool
def compute_derivative(expression: str, variable: str) -> str:
    """Computes the derivative of a mathematical expression with respect to a variable"""
    from sympy import symbols, diff, sympify

    var = symbols(variable)
    expr = sympify(expression)
    derivative = diff(expr, var)

    return f"The derivative of {expression} with respect to {variable} is {derivative}"


# Tool registry for dispatch
TOOLS = [get_weather, calculate_geometric_function, count_character_in_string, compute_derivative]
TOOL_MAP = {t.name: t for t in TOOLS}


# ============================================
# PART 2: State Definition
# ============================================

class AgentState(TypedDict):
    messages: List[Any]       # Full LangChain message history
    should_exit: bool         # Whether the user wants to quit
    user_input: str           # Latest raw user input


# ============================================
# PART 3: Create LLM with Tools
# ============================================

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
llm_with_tools = llm.bind_tools(TOOLS)


# ============================================
# PART 4: Graph Nodes and Routing
# ============================================

def get_user_input(state: AgentState) -> dict:
    """Node that prompts the user for input via stdin."""
    print("\n" + "=" * 60)
    print("Enter your message (or 'quit' to exit):")
    print("=" * 60)
    print("\n> ", end="")
    raw = input().strip()

    if raw.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        return {"user_input": raw, "should_exit": True}

    # Append the user message to conversation history
    messages = list(state.get("messages", []))
    messages.append(HumanMessage(content=raw))

    return {
        "user_input": raw,
        "should_exit": False,
        "messages": messages,
    }


def call_llm(state: AgentState) -> dict:
    """Node that sends the conversation to the LLM (with tools bound)."""
    messages = state["messages"]

    print("\nThinking...")
    response = llm_with_tools.invoke(messages)

    # Append the assistant response to message history
    messages = list(messages)
    messages.append(response)

    return {"messages": messages}


def execute_tools(state: AgentState) -> dict:
    """Node that executes any tool calls the LLM requested."""
    messages = list(state["messages"])
    # The last message is the AIMessage with tool_calls
    ai_message = messages[-1]

    for tool_call in ai_message.tool_calls:
        function_name = tool_call["name"]
        function_args = tool_call["args"]

        print(f"  Tool: {function_name}")
        print(f"  Args: {function_args}")

        # Dispatch using the tool registry
        if function_name in TOOL_MAP:
            result = TOOL_MAP[function_name].invoke(function_args)
        else:
            result = f"Error: Unknown function {function_name}"

        print(f"  Result: {result}")

        messages.append(ToolMessage(
            content=result,
            tool_call_id=tool_call["id"]
        ))

    return {"messages": messages}


def print_response(state: AgentState) -> dict:
    """Node that prints the final assistant response."""
    # The last message should be the AIMessage with the final answer
    ai_message = state["messages"][-1]
    print("\n" + "-" * 60)
    print("Assistant:")
    print("-" * 60)
    print(ai_message.content)
    return {}


def route_after_input(state: AgentState) -> str:
    """Route after user input: quit or continue to LLM."""
    if state.get("should_exit", False):
        return END
    if state.get("user_input", "").strip() == "":
        return "get_user_input"
    return "call_llm"


def route_after_tools(state: AgentState) -> str:
    """
    Route after tool execution or LLM response.
    If the last message is an AIMessage with tool_calls, execute tools.
    If the last message has tool results, send back to LLM for final answer.
    Otherwise, print the response.
    """
    last_message = state["messages"][-1]

    # If the LLM wants to call tools, go execute them
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "execute_tools"

    # If we just got tool results back, let the LLM process them
    if isinstance(last_message, ToolMessage):
        return "call_llm"

    # Otherwise the LLM gave a final text answer
    return "print_response"


# ============================================
# PART 5: Build the Graph
# ============================================

def create_graph(db_path: str = "tool_agent_checkpoints.db"):
    """
    Build the LangGraph with checkpointing.

    Graph flow:
        START -> get_user_input -> [route_after_input] -> call_llm -> [route_after_tools]
                      ^                    |                                |     |
                      |                   END              execute_tools <--+     |
                      |                                         |                |
                      |                                    [route_after_tools]    |
                      |                                         |                |
                      |                                    call_llm (loop)       |
                      |                                                          |
                      +------- print_response <----------------------------------+
    """
    graph_builder = StateGraph(AgentState)

    # Add nodes
    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_llm", call_llm)
    graph_builder.add_node("execute_tools", execute_tools)
    graph_builder.add_node("print_response", print_response)

    # START -> get_user_input
    graph_builder.add_edge(START, "get_user_input")

    # get_user_input -> route to call_llm or END
    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "call_llm": "call_llm",
            "get_user_input": "get_user_input",
            END: END,
        },
    )

    # call_llm -> route: execute_tools (if tool calls) or print_response (final answer)
    graph_builder.add_conditional_edges(
        "call_llm",
        route_after_tools,
        {
            "execute_tools": "execute_tools",
            "print_response": "print_response",
        },
    )

    # execute_tools -> route: back to call_llm (LLM needs to see results)
    graph_builder.add_conditional_edges(
        "execute_tools",
        route_after_tools,
        {
            "call_llm": "call_llm",
        },
    )

    # print_response -> loop back to get_user_input
    graph_builder.add_edge("print_response", "get_user_input")

    # Set up SQLite checkpointing for recovery
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    graph = graph_builder.compile(checkpointer=checkpointer)
    return graph, conn


# ============================================
# PART 6: Run
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("LangGraph Tool-Calling Agent with Checkpointing")
    print("=" * 60)

    graph, conn = create_graph(db_path="tool_agent_checkpoints.db")

    try:
        # Save graph visualization
        try:
            png_data = graph.get_graph(xray=True).draw_mermaid_png()
            with open("tool_agent_graph.png", "wb") as f:
                f.write(png_data)
            print("Graph image saved to tool_agent_graph.png")
        except Exception as e:
            print(f"Could not save graph image: {e}")

        # Invoke with a thread_id so checkpointing can track this session.
        # On first run, state is initialized fresh.
        # On subsequent runs with the same thread_id, state is restored from checkpoint.
        graph.invoke(
            {
                "messages": [
                    SystemMessage(content="You are a helpful assistant. Use the provided tools when needed.")
                ],
                "should_exit": False,
                "user_input": "",
            },
            config={"configurable": {"thread_id": "session-1"}},
        )
    finally:
        conn.close()
