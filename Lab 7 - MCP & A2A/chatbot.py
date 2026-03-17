import os
import json
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

SYSTEM_PROMPT = (
    "You are a Semantic Scholar research assistant with access to the Asta MCP tools. "
    "Use these tools to find academic papers, trace citations, and explore research topics. "
    "Always cite paper titles and years in your responses. If a tool call fails, "
    "acknowledge the error and try an alternative approach if possible."
)


def call_mcp(method, params):
    """Send a JSON-RPC 2.0 request to Asta and parse the SSE response."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params,
    }
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


def get_asta_tools():
    """Fetch tool schemas from MCP and convert to OpenAI function format."""
    result = call_mcp("tools/list", {})

    # tools/list result may be double-JSON-encoded
    if isinstance(result, str):
        result = json.loads(result)

    mcp_tools = result.get("tools", [])

    openai_tools = []
    for tool in mcp_tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("inputSchema", {"type": "object", "properties": {}}),
            },
        })
    return openai_tools


def call_asta_tool(name, arguments):
    """Execute a tools/call and return the text content."""
    try:
        mcp_result = call_mcp("tools/call", {"name": name, "arguments": arguments})
    except Exception as e:
        return f"Error calling tool {name}: {e}"

    if mcp_result.get("isError"):
        content = mcp_result.get("content", [{}])
        return f"Error: {content[0].get('text', 'Unknown error')}"

    # Prefer structuredContent — already parsed, no double-decode needed
    structured = mcp_result.get("structuredContent")
    if structured is not None:
        return json.dumps(structured, indent=2)

    # Fall back: join all text content items
    content = mcp_result.get("content", [])
    texts = [item["text"] for item in content if item.get("type") == "text"]
    return "\n".join(texts)


def chat(user_message, messages, tools):
    """One turn of the chatbot loop, handling tool calls until a final answer."""
    messages.append({"role": "user", "content": user_message})

    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        message = response.choices[0].message

        # Append assistant message as a plain dict (SDK objects aren't re-serializable)
        assistant_msg = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        messages.append(assistant_msg)

        if not message.tool_calls:
            return message.content

        # Execute each tool call and append results
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"  [tool] {name}({json.dumps(arguments)})")
            result = call_asta_tool(name, arguments)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })


def main():
    print("Fetching tool schemas from Asta MCP server...")
    tools = get_asta_tools()
    print(f"Loaded {len(tools)} tools: {[t['function']['name'] for t in tools]}\n")

    print("Asta Research Chatbot — type 'quit' to exit")
    print("Test queries:")
    print('  "Find recent papers about large language model agents"')
    print('  "Who wrote Attention is All You Need and what else have they published?"')
    print('  "What papers cite the original BERT paper?"')
    print('  "Summarize the references used in the ReAct paper"\n')

    # Conversation history persists across turns
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        answer = chat(user_input, messages, tools)
        print(f"Assistant: {answer}\n")


if __name__ == "__main__":
    main()
