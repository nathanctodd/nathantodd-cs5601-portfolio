# File pulls list of all tools from Asta MCP endpoint and prints their names, descriptions, and input parameters.

# Question: Which tool would you use to find all papers about "transformer attention mechanisms"? 
# Answer: I would use the search_paper_by_title tool, or search_papers_by_relevance tool to find papers about 
#         "transformer attention mechanisms".

# Question: Which would you use to find who else published in the same area as a specific author?
# Answer: I would use maybe consider using get_citations and then look to see who cited it and the authors 
#         that cited it. I could also use search_papers_by_relevance to find papers in the same area and then 
#         look at the authors of those papers.


import requests
import json
import os

# Asta MCP endpoint
ASTA_ENDPOINT = "https://asta-tools.allen.ai/mcp/v1"  # Adjust URL as needed

# Get API key from environment
API_KEY = os.environ.get("ASTA_API_KEY")
if not API_KEY:
    raise ValueError("ASTA_API_KEY environment variable not set")

# JSON-RPC 2.0 request for tools/list
payload = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list",
    "params": {}
}

# Send POST request with API key in headers
response = requests.post(
    ASTA_ENDPOINT,
    json=payload,
    headers={
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "Authorization": f"Bearer {API_KEY}"
    }
)

# Print response
print("Status Code:", response.status_code)

if response.status_code == 200:
    try:
        # Parse Server-Sent Events (SSE) format
        for line in response.text.strip().split('\n'):
            if line.startswith('data: '):
                json_data = line[6:]  # Remove 'data: ' prefix
                parsed = json.loads(json_data)
                result = parsed.get('result', [])
                # Handle case where result is a string that needs parsing
                if isinstance(result, str):
                    result = json.loads(result)
                result = json.loads(result) if isinstance(result, str) else result
                tools = result.get('tools', [])
                for tool in tools:
                    print(f"Tool: {tool['name']}")
                    tool_desc = tool.get('description', 'No description available').split('\n')[1]
                    print(f"  Description: {tool_desc}")
                    
                    input_schema = tool.get('inputSchema', {})
                    properties = input_schema.get('properties', {})
                    required_params = input_schema.get('required', [])
                    
                    # Print required parameters
                    if required_params:
                        req_info = []
                        for param in required_params:
                            param_type = properties.get(param, {}).get('type', 'unknown')
                            req_info.append(f"{param} ({param_type})")
                        print(f"  Required: {', '.join(req_info)}")
                    
                    # Print optional parameters
                    optional_params = [p for p in properties.keys() if p not in required_params]
                    if optional_params:
                        opt_info = []
                        for param in optional_params:
                            param_type = properties.get(param, {}).get('type', 'unknown')
                            opt_info.append(f"{param} ({param_type})")
                        print(f"  Optional: {', '.join(opt_info)}")
                    
                    print()
    except (json.JSONDecodeError, ValueError) as e:
        print("Error parsing response:", e)
        print("Raw response:", response.text)
else:
    print("Error: Received status code", response.status_code)
    print("Response Text:", response.text)
    
    
    
    
    



