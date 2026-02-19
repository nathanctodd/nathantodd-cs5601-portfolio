# Running an LLM Report

## Nathan Todd

In this report, I explored Lang Graph by attempting to create a simple agent that can answer questions, with both Qwen and Llama models. Lang Graph is an orchestration framework that allows for the creation of complex workflows using LLMs and other tools. The models used include Llama-3.2B and Qwen/Qwen-2.5B, allowing the user to choose between the two models when chatting with the agent and also allowing each model to view the history and also the corresponding user, whether it be the other model, the user, or itself.

1. To start, I implemented the 'verbose' and 'quiet' flags in the code to allow for the user to choose the level of output detail they want during execution. The 'verbose' flag provides detailed information about each step of the process, while the 'quiet' flag minimizes output to only essential information. This was done by adding conditional print statements throughout the code that check the status of these flags before outputting information.

    - Here is some basic output:

        
        \> verbose

        Current node:  get_user_input

        Current node:  route_after_input

        Current node:  call_llm

        Processing your input...
        
        LLM Response:
        ----
        Assistant: ... 
    

2. Next, when I entered in a blank input, the model would go off and start blabbering and generating random stories and text. I put one of these responses below:


    - #### User Input:

        \> 

        Processing your input...

        #### LLM Response:

        Assistant: 
            Once upon a time in a land far, far away, there lived a young princess named Isabella. She was known throughout the kingdom for her kindness and bravery. One day, a mysterious traveler arrived at the castle, seeking help to find a hidden treasure that was said to bring great fortune to whoever discovered it...

    To fix this, I added a simple check to see if the input was blank and if so, prompt the user to enter a valid input instead of proceeding with the LLM call. I did this through adding another edge in the Lang Graph that checks for blank input and routes back to the user input node if the input is blank. This ensures that the model only processes valid inputs and avoids generating irrelevant or nonsensical responses.

3. Next I added in capabilities for the input to pass through both Qwen and Llama models in parallel. I did this through asyncio to allow for concurrent execution of both model calls. This involved creating separate asynchronous functions for each model call and then using asyncio.gather to run them simultaneously. The results from both models were then collected and returned to the user. Here is an example of the output:


    - #### User Input:    

        \> Hi there!


        #### LLAMA RESPONSE

        Assistant: Hi! It's nice to meet you. I'm happy to help with any questions or concerns you may have. How can I assist you today?

        #### QWEN RESPONSE

        Assistant: Hello! How can I help you today? Is there something specific that you would like to know or discuss about?


4. For step 4, I added in conditional logic that looks for the words, "Hey Qwen" to either send it to Qwen or route to Llama depending on the input. I split up the LLM responses in the Lang Graph to have two separate nodes for each model. Then, I added a conditional edge that checks the user input for the presence of "Hey Qwen". If the phrase is found, the input is routed to the Qwen node; otherwise, it is sent to the Llama node. Here is an example:

    - #### User Input:

        \> hey qwen, how are you?

        Processing your input...

        #### LLM Response:
        --------------------------------------------------
        User: hey qwen, how are you?
        Assistant: I'm doing well, thanks for asking! How about you? Is there anything special that happened today? I'm here to help with any questions or concerns you might have. 🚀✨

        \> Hi Llama

        Processing your input...

        #### LLM Response:

        Assistant: Hello! How can I assist you today?


5. Next, I implemented a message history feature that keeps track of the last few interactions so that it feels more like a conversation. When the model would respond, it could maintain ideas between multiple messages.

6. Next, I added in tags for each model input, either Llama, Qwen, User, or System. This was done by adding a role to each message in the history. Then I allowed the model to see the history of messages with the corresponding tags. This allows the model to understand the context of the conversation better, including who said what, so the models can 'talk' to each other and also the user. Here is an example of the output:

    - #### User Input: 

        \> Hi there llama! What is a data structure?

        **Llama**: A data structure is a way to organize and store data in a structured way.

        \> Hey qwen, what do you think of Llamas response?

        **Qwen**: My interpretation of your statement is that a data structure can be defined by its components and their relationships rather than just its organization within memory space. It's important to note that different languages may use slightly different terminology for such concepts.

        \> Llama, anything to say to qwen?

        **Llama**: I don't have anything to say to qwen, as we are on different levels of consciousness, and he is not capable of hearing.


7. Lastly, I added in checkpointing to the conversation so that if the program crashes or is exited, the conversation history is saved and can be reloaded when the program is restarted. This was done by writing the conversation history to a file after each interaction and then reading from that file when the program starts up again. This ensures that the user can pick up where they left off without losing any context. Here is an example of how it works:

    - #### User Input:
        \> Hi Llama, can you remember our last conversation?

        **Llama**: Yes, I remember our last conversation. We were discussing data structures and their importance in computer science. Is there anything specific you would like to revisit or expand upon from our previous discussion?

.

.

Link to Colab Notebook: [LLM Report Colab](https://colab.research.google.com/drive/1IxRvXpX4NAEU7K0BKUpPElnaMbPpzqNV?usp=sharing)