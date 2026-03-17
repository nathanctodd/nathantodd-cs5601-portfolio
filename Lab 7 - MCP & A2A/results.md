# A2A Discovery and Task Routing


## Discussion Questions


1. MCP vs A2A: How is sending a task to another agent different from calling an MCP tool? What can an agent do that a tool cannot?
    - MCP (Model Call Protocol) is a standardized way for agents to call tools, which are typically external APIs or functions that perform specific tasks. A2A (Agent-to-Agent) communication, on the other hand, involves one agent sending a task or request directly to another agent, which may have its own capabilities and knowledge.
    - An agent can perform more complex reasoning, maintain state, and have its own goals and strategies, while a tool is usually a stateless function that performs a specific action. For example, an agent could decide to call multiple tools, aggregate their results, and make decisions based on that, while a tool would just execute its function and return a result.

2. Discovery: We used a central registry. What are the alternatives? What are the tradeoffs of centralized vs decentralized discovery?
    - Alternatives to a central registry include decentralized peer-to-peer discovery, where agents broadcast their capabilities and listen for others, or using a distributed ledger or blockchain to register and discover agents.
    - Centralized discovery can be simpler to implement and manage, but it can become a bottleneck and a single point of failure. Decentralized discovery can be more robust and scalable, but it may require more complex protocols for communication and trust.

3. System prompts as strategy: How much did the system prompt matter for scoring? Could you craft a prompt that is good at all categories while still being funny on off-topic questions?
    - The system prompt can significantly influence the behavior of the agents and how they score tasks. A well-crafted prompt can guide agents to provide more accurate and relevant responses, while a poorly designed prompt may lead to confusion or irrelevant answers.
    - It is possible to craft a prompt that is effective across multiple categories while also being humorous for off-topic questions. This would require careful wording to ensure that the prompt encourages agents to focus on the relevant information while also allowing for some creativity in handling unexpected or off-topic queries.

4. Smart routing: TF-IDF matched questions to agents based on text overlap. What would happen with semantic embeddings instead? What if agents could self-report confidence?
    - Using semantic embeddings instead of TF-IDF could allow for better matching of questions to agents based on the meaning of the text rather than just keyword overlap. This could improve routing for questions that are phrased differently but have the same intent.
    - If agents could self-report confidence, the system could use this information to route tasks more effectively. For example, if an agent reports low confidence in its ability to answer a question, the system could choose to route that question to a different agent with higher confidence, potentially improving overall performance.

5. Trust and reliability: In a real multi-agent system, how would you handle an agent that returns bad data? What if an agent is slow or goes offline mid-task?
    - To handle an agent that returns bad data, the system could implement a validation step where the output of the agent is checked against certain criteria or compared with outputs from other agents. If an agent consistently returns bad data, it could be flagged for review or removed from the system.
    - If an agent is slow or goes offline mid-task, the system could implement timeouts and retries. If an agent does not respond within a certain time frame, the system could attempt to route the task to another agent or return an error message to the user. Additionally, the system could keep track of agent performance and reliability over time to make informed decisions about routing tasks.

6. Scaling: What would break if there were 1,000 agents instead of 20? What architectural changes would you need?
    - With 1,000 agents, the central registry could become a bottleneck for discovery and routing. The system would need to implement more efficient indexing and search mechanisms to handle the increased number of agents.
    - Architectural changes might include implementing a distributed registry or using a more scalable database for storing agent information. Additionally, the routing algorithm would need to be optimized to quickly match tasks to the most appropriate agents without having to check each one individually. This could involve using machine learning models to predict which agents are best suited for certain tasks based on historical performance and capabilities.


## Output from Terminal

```bash
Task endpoint: https://nonalcoholic-pam-livingly.ngrok-free.dev/task
Skills: Movie Q&A, TV Show Q&A, Movie Recommendations, Cast Information, Plot Summaries

Ready to receive tasks!

INFO:     Started server process [81157]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     199.111.224.18:0 - "GET /health HTTP/1.1" 200 OK
INFO:     199.111.224.45:0 - "GET /.well-known/agent.json HTTP/1.1" 200 OK
INFO:     199.111.224.18:0 - "GET /health HTTP/1.1" 200 OK
INFO:     199.111.224.18:0 - "GET /health HTTP/1.1" 200 OK
INFO:     199.111.224.18:0 - "GET /health HTTP/1.1" 200 OK
INFO:     199.111.224.18:0 - "GET /health HTTP/1.1" 200 OK
INFO:     199.111.224.18:0 - "GET /health HTTP/1.1" 200 OK
INFO:     199.111.224.18:0 - "GET /health HTTP/1.1" 200 OK

Received task from trivia-master: In what year did the United States Women's National Team win their first FIFA Women's World Cup?
Responding: The United States Women's National Team won their first FIFA Women's World Cup in 1991. The tourname...
INFO:     199.111.224.18:0 - "POST /task HTTP/1.1" 200 OK
INFO:     199.111.224.18:0 - "GET /health HTTP/1.1" 200 OK

Received task from trivia-master: In what year did the United States Women's National Team win their first FIFA Women's World Cup?
Responding: The United States Women's National Team won their first FIFA Women's World Cup in 1991. The tourname...
INFO:     199.111.224.18:0 - "POST /task HTTP/1.1" 200 OK
INFO:     199.111.224.18:0 - "GET /health HTTP/1.1" 200 OK

Received task from trivia-master: What is the only country to have played in every FIFA Men's World Cup tournament?
Responding: The only country to have played in every FIFA Men's World Cup tournament since the inaugural event i...
INFO:     199.111.224.18:0 - "POST /task HTTP/1.1" 200 OK

Received task from trivia-master: In basketball, how many points is a shot made from behind the three-point line worth?
Responding: In basketball, a shot made from behind the three-point line is worth three points....
INFO:     199.111.224.18:0 - "POST /task HTTP/1.1" 200 OK

Received task from trivia-master: In what year did the United States Women's National Team win their first FIFA Women's World Cup?
Responding: The United States Women's National Team won their first FIFA Women's World Cup in 1991. The tourname...
INFO:     199.111.224.18:0 - "POST /task HTTP/1.1" 200 OK
INFO:     199.111.224.18:0 - "GET /health HTTP/1.1" 200 OK
INFO:     199.111.224.18:0 - "GET /health HTTP/1.1" 200 OK
```