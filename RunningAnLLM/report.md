# Running an LLM Report

In this report, I first utilized various LLMs to generate responses and see how they perform specifically on the MMLU benchmark. All of the LLMs used were found on Hugging Face. The models used include Llama-3.2B, Qwen/Qwen-2.5B, and allenai/OLMo-2-0425. 

1. - 2. To start, I setup the environment by installing the necessary libraries, including the Hugging Face Transformers library and then also authorized access to the models and hugginface API using my token.

3. - 4. Next, I utilized the given script to run the various models on the MMLU benchmark. I first began only with the Llama-3.2B model to ensure that everything was working correctly. I ran the model on 2 topics from the MMLU benchmark (astronomy and business ethics) and recorded both the accuracy of the model as well as the time taken to generate the responses. The results are listed below.

GPU without Quantization:
- Astronomy: 50.00%, Business Ethics: 45.00%
- Time Taken: 113 seconds (1.88 minutes)

CPU without Quantization:
- Astronomy: 50.00%, Business Ethics: 45.00%
- Time Taken: 8567 seconds (2.34 hours)

GPU with 4-bit Quantization:
- Astronomy: 45.00%, Business Ethics: 40.00%
- Time Taken: 527 seconds (8.78 minutes)


This illustrates the significant difference in time taken between using a GPU and CPU, as well as the impact of quantization on both time and accuracy. The GPU runs were significantly faster than the CPU runs, and quantization reduced the time taken but also slightly decreased accuracy. This is expected, as quantization often leads to a trade-off between speed and performance.


5. Next, I successfully adjusted the code to include a small flag to print out the questions and their answers as well as including the time taken, CPU time, and GPU time in the model summary at the end of the run. This was done by simply adding print statements in the appropriate sections of the code and using the time library to track the time taken for each response. I ran the model again on 10 topics and observed the output.

6. I created graphs of the results of the 10 topics run with Llama-3.2B and Qwen/Qwen-2.5B both on GPU without quantization. The graphs show the accuracy of each of the topics for both models. The graphs are included in the graphs_code.pdf. The results show that Qwen performed better than Llama03.2B on the college physics and computer science topics, while Llama-3.2B performed better on the other topics. This makes sense, as Llama has more parameters and is generally a more powerful model, but still interesting that there was such a significant difference in those two topics. Qwen saw a 10% increase in accuracy on college physics and a 13% increase on computer science compared to Llama-3.2B. Both models didn't do outstanding though, with most topics being below 50% accuracy. This is to be expected with models of this size, as MMLU is a challenging benchmark and smaller models often struggle to perform well on it.

7. The above experiments were repeated in Google Colab with stronger resources. The results can be found here: [Google Colab Results](https://colab.research.google.com/drive/1L2MovpF0KWv8Yhe4TP5xnKnAGLNLPT1M?usp=sharing). The results were similar to those obtained on my local machine, with Qwen/Qwen-2.5B generally outperforming Llama-3.2B on the MMLU benchmark. The time taken for each model was significantly reduced due to the stronger hardware available in Colab, allowing for faster inference times. For example, Qwen was able to complete 10 topic runs is just 2 minutes on Colab, compared to about 2 minutes for just 2 topics on my local GPU. Llama took just over 2.3 minutes for 10 topics on Colab. However, both the Colab GPU and local GPU far outperformed the CPU runs, which took hours to complete without quantization.

8. To create a chat agent, I utilized the pre-created script that was given and just made small adjustments. To avoid the model from overrunning the context windows, I implemented a simple rolling window method to simply drop the oldest messages once the context window limit was reached. This ensures that the most recent messages are always included in the context, allowing for more relevant responses from the model. I tested the chat agent with Llama 3.2B and it works quite well and doesn't overrun the context window. In addition, I also added a flag to delete the context window on every new user input, so that the model never retains information about the conversation and always starts fresh. This can be useful in scenarios where privacy is a concern or when the user wants to ensure that previous conversations do not influence the current interaction by making each new message independent. Overall, the chat agent functions effectively with these adjustments, providing coherent and contextually relevant responses while managing the context window appropriately. The chat can be found here: [Chat Agent Colab](https://colab.research.google.com/drive/1nqDm53GItlyoatCslqnFxXXpBguhwR6v?usp=sharing).



