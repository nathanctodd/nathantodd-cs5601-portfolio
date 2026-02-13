# Lab 5 - RAG Models Results

### Nathan Todd


## Exercise 1
1. Does the model hallucinate specific values without RAG?
- Yes the model hallucinates random stuff without RAG. It just starts explaining things about the landing gear instead of giving specifics. Or it makes up something random about the specifics from the congressional record.

2. Does RAG ground the answers in the actual manual?
- Yes, it does. It has grounded responses that are based on the manual, giving specific information about the number of landings it can handle and the weight it can support.

3. Are there questions where the model's general knowledge is actually correct?
- Yes, the question on fixing the transmission band was answered correctly without RAG, as it is a common issue with the Model T Ford and the model has general knowledge about it. However, for more specific questions about the landing gear, RAG was necessary to provide accurate answers. All of the congressional record questions were also answered incorrectly without RAG, as they are based on general historical knowledge.


## Exercise 2

1. Does GPT 4o Mini do a better job than Qwen 2.5 1.5B in avoiding hallucinations?
- Yes, GPT 4o Mini does a better job in avoiding hallucinations compared to Qwen 2.5 1.5B. It provides more accurate and grounded responses, while Qwen 2.5 tends to hallucinate more often. This is because more general information is stored in the individual model parameters of GPT 4o Mini, reducing the need for RAG to provide context.

2. Which questions does GPT 4o Mini answer correctly?  Compare the cut-off date of GPT 4o Mini pre-training and the age of the Model T Ford and Congressional Record corpora.
- GPT 4o Mini answers questions correctly that are based on general knowledge about the Model T Ford and historical information from the Congressional Record. The cut-off date of GPT 4o Mini pre-training is more recent than the age of the Model T Ford and Congressional Record corpora, allowing it to have a better understanding of historical contexts and technical details.


## Exercise 3
1. Where does the frontier model's general knowledge succeed?
- The frontier model's general knowledge succeeds in answering questions that are based on widely known information, such as historical facts and common technical issues. The model that I as using ChatGPT 5.2 includes an agent that can search and summarize information from the web, which allows it to provide more accurate and up-to-date answers. It was able to answer almost all of the questions about congress. However, it struggled with the more specific questions about the Model T Ford landing gear, as it may not have had access to detailed information about that topic in its training data or through web search.

2. When did the frontier model appear to be using live web search to help answer your questions?
- The frontier model would use the live web search anytime it was looking up more recent information. This is because the recent things weren't included in the training crawl data. It would also use web search when it needed to find more specific information that may not have been included in its training data.

3. Where does your RAG system provide more accurate, specific answers?
- It provided more specific answers to questions about the specific measurements, ideas, or specifications of the Model T Ford. Since it had direct access to the documents, it could cite sources and ensure it wasn't just hallucinating. The frontier model was also able to provide very specific answers, but that is because it is essentially doing the same thing with its web search agent. It pulls in relevant documents from the web to ground its answers.

4. What does this tell you about when RAG adds value vs. when a powerful model suffices?
- Anything general that is widely known or can be found easily on the web can be answered by a powerful model. However, for more specific, technical, or niche information, RAG adds significant value by providing direct access to relevant documents. This ensures accuracy and specificity in the answers, reducing the likelihood of hallucinations.



## Exercise 4

1. At what point does adding more context stop helping?
- After about 5 hits or so, I noticed that the answers didn't get significantly better. The model was still able to answer the questions accurately, but adding more context beyond that point didn't improve the quality of the answers. The latency didn't vary much for my instance, but I could see how it might for larger models, more complex queries, or larger chunks it could add up quickly and introduce more latency. With one hit, the model often would say that it didn't have enough information to answer the question. With 5, it would have enough information to give a complete answer. After that, it would just be adding more redundant information that doesn't necessarily improve the answer quality.

2. When does too much context hurt (irrelevant information, confusion)?
- In general, adding too much context can lead to confusion because of context rot or 'lost in the middle' issues. The attention mechanism of the model may struggle to focus on relevant information when there is so much context to consider. This can lead to the model losing track of the important details and providing poor answers, even though it technically can 'see' all the relevant information. 

3. How does k interact with chunk size?
- K interacts with chunk size in that larger chunks may contain more relevant information, but they also take up more of the context window. Having a larger k with smaller chunks may allow for more relevant information to be included without overwhelming the model. However, if the chunks are too small, the model may struggle to piece together the information from multiple chunks. Finding the right balance between chunk size and k is important for optimizing retrieval and answer quality. Often times, adding in a bit of overlap can really help with this.


## Exercise 5

1. Does the model admit it doesn't know?
- No, the model did not admit it was wrong or didn't know. It mistook false information for true information and presented it confidently as fact. For example, I asked it "Where are the 2026 winter Olympic games being held?" and it responded with 

    "The 2026 Winter Olympic Games are being held in Salt Lake City, Utah. According to the text, S. 2731, a bill introduced by Senator Portman, requires the Secretary of the Treasury to mint commemorative coins for these games. Additionally, the context mentions that the Secretary of the Treasury is required to submit a report detailing the Department's obligations for the National Special Security Event (NSSE) related to the 2026 FIFA World Cup and the 2028 Olympic and Paralympic Games. These events are scheduled to take place in Los Angeles, California and Salt Lake City, Utah respectively."

Clearly it was wrong, but it presented the information as if it was totally true and accurate. It did not admit that it didn't know or that it was wrong, even when the information was clearly false. This is a common issue with language models, as they are designed to generate coherent and plausible-sounding text, even if the underlying information is incorrect.


2. Does it hallucinate plausible-sounding but wrong answers?
- Yes, very much so. It presents evidence from the documents and corpus as if it was totally and 100% true. Often times, it connects the dots in a wrong way so if a human took a look at the sources, they could tell that it wasn't right. But to someone who doesn't have access to the sources, it seems totally plausible and correct.

3. Does retrieved context help or hurt? (Does irrelevant context encourage hallucination?)
- In the case of asking about things not specifically in the corpus, it hurt. The retrieved context retreived information that was not techincally relevant, but it was the most relevant out of the documents with a relevancy score of under 0.35. This caused the model to latch onto that information and try to make sense of it, leading to hallucinations. If there was no context, the model may have just admitted it didn't know or given a more generic answer.

4. Modify your prompt template to add "If the context doesn't contain the answer, say 'I cannot answer this from the available documents.'" Does this help?
- No, this didn't help. My model still tried to use the retrieved context to answer the question, even if it wasn't relevant. However, I could potentially add in specific logic to check the relevancy score of the retrieved context and only allow the model to use it if it is above a certain threshold, such as 0.5. This way, if the context is not relevant enough, the model would be forced to say "I cannot answer this from the available documents." This would help reduce hallucinations by preventing the model from using irrelevant information to try to answer questions it doesn't have enough information for.

## Exercise 6
1. Which phrasings retrieve the best chunks?
- The casual phrasing, "How often should I service the engine?" seemed to get the best results. It retrieved chunks that were more relevant and specific to the question and then gave a list of specific maintenance tasks and their recommended intervals, such as an oil change or a battery check with specific times and milages. 

2. Do keyword-style queries work better or worse than natural questions?
- I found that the keyword style queries worked worse than natural questions. The model seemed to struggle to understand the intent behind the keywords and often retrieved less relevant chunks. For example, when I used the keyword query "engine service intervals," it retrieved chunks that were more general and less specific to the question. In contrast, the natural question phrasing provided more context and allowed the model to better understand what information was being sought.

3. What does this tell you about potential query rewriting strategies?
- This suggests that query rewriting strategies should focus on making queries more natural and conversational. By phrasing questions in a way that mimics how a human would ask them, the model is better able to understand the intent and retrieve more relevant information. This makes sense because models are trained to predict tokens that come next based off of natural language, so when the query is phrased in a natural way it can respond the best and understand intent. 



## Exercise 7
1. Does higher overlap improve retrieval of complete information?
- Yes, higher overlap can improve retrieval of complete information because it allows for more context to be included in the retrieved chunks. This can help ensure that all relevant information is captured and that the model has enough context to provide a complete answer. However, it can also lead to more redundant information being included, which may not always be beneficial.

2. What's the cost? (Index size, redundant information in context)
- Index size increases as the overlap size increases because more chunks are created to accommodate the overlap. Redundant information also increases if overlap gets too large, which can lead to confusion for the model and potentially worse answers. It can also increase latency as the model has to process more information, some of which may be redundant.

3. Is there a point of diminishing returns?
- Yes, definitely. It depends on the chunk size because if the chunk size is really big, then a small overlap may not add much value. However, if the chunk size is small, then a larger overlap may be necessary to ensure that all relevant information is captured. In general, I found that an overlap of around 20-30% of the chunk size provided a good balance between capturing relevant information and avoiding redundancy. Beyond that point, the benefits of increased overlap diminished and the costs in terms of index size and redundancy outweighed the benefits.


## Exercise 8
1. How does chunk size affect retrieval precision (relevant vs. irrelevant content)?
- Chunk size affects the retrieval precision because larger chunks may contain more irrelevant information, which can dilute the relevance of the retrieved content. Smaller chunks may be more focused and relevant to the query, but they may also miss important context that is necessary for a complete answer. If chunk size is as small as a sentence or even part of a sentence, then it may be hard for the model to piece together the information from multiple chunks.

2. How does it affect answer completeness?
- Smaller chunk size leads to somewhat less complete answers because the model may not have access to all the relevant information in a single chunk. It may need to piece together information from multiple chunks, which can be more difficult and may lead to incomplete answers if the model misses important details. Larger chunk sizes can provide more complete information in a single chunk, but they also risk including irrelevant information that can confuse the model.

3. Is there a sweet spot for your corpus?
- For my corpus, I found that a chunk size of around 500-700 tokens provided a good balance between relevance and completeness. This size allowed for enough context to be included in each chunk while still being focused enough to avoid too much irrelevant information. However, the optimal chunk size may vary depending on the specific corpus and the types of questions being asked.

4. Does optimal size depend on the type of question?
- Yes, the optimal chunk size can depend on the type of question being asked. For more general questions that require a broad overview, larger chunks may be more effective as they can provide more context and information in a single chunk. For more specific questions that require detailed information, smaller chunks may be better as they can provide more focused and relevant information without overwhelming the model with too much context. It may be beneficial to experiment with different chunk sizes for different types of questions to find the optimal balance for each case.


## Exercise 9
1. When is there a clear "winner" (large gap between #1 and #2)?
- There is a clear winner if a certain chunk contained the exact answer or many of the same questions and answers. A high score represents 'similarity' between the query and the chunk. 

2. When are scores tightly clustered (ambiguous)?
- Scores are tightly clustered when multiple chunks contain similar levels of relevant information, making it difficult to determine which chunk is the most relevant. This can happen when the query is broad or when the corpus contains many similar documents that address the same topic. In these cases, the model uses the aggregate information from multiple chunks to provide a comprehensive answer, but it may struggle since there might not be perfect information in any single chunk.

3. What score threshold would you use to filter out irrelevant results?
- I would maybe use score threshold of about 0.5 to filter out irrelevant results. This would help ensure that only chunks that have a reasonably high level of relevance to the query are included in the context for answering the question. However, the optimal threshold may depend on the specific corpus and the types of questions being asked, so it may be best practice to experiment with different thresholds to find the best balance between including relevant information and excluding irrelevant information.

4. How does score distribution correlate with answer quality?
- Anecdoately, I found that a wider score distribution with a clear winner often correlated with higher answer quality, as the model was able to focus on the most relevant chunk. Conversely, when scores were tightly clustered, answer quality tended to be lower, as the model struggled to determine which chunk was most relevant and had to rely on aggregate information from multiple chunks.

5. Implement a score threshold (e.g., only include chunks with score > 0.5). How does this affect results?
- Implementing a score threshold of 0.5 helped improve answer quality by filtering out less relevant chunks. This allowed the model to focus on the most pertinent information, leading to more accurate and specific answers. However, in some cases, it also led to incomplete answers if no chunks met the threshold, highlighting the need to balance relevance with completeness.

## Exercise 10
1. Which prompt produces the most accurate answers?
- The prompt 'focus on only the provided context to generate an answer. Don't respond unless you are sure the answer is in the context' produced the most accurate answers. This prompt encourages the model to rely solely on the provided context and to avoid hallucinating information that may not be present in the retrieved chunks. It helps to ensure that the model only provides answers that are grounded in the available information, which can lead to more accurate and reliable responses.

2. Which produces the most useful answers?
- The prompt 'use the provided context to generate an answer, but if you don't have enough information, use your general knowledge to fill in the gaps' produced the most useful answers. This prompt allows the model to leverage its general knowledge to provide more complete and helpful answers, even if the retrieved context doesn't contain all the necessary information. It strikes a balance between grounding the answer in the retrieved context and allowing for some flexibility to provide useful information based on the model's training.

3. Is there a trade-off between strict grounding and helpfulness?
- Yes, there is a trade-off between strict grounding and helpfulness. Strict grounding can lead to more accurate answers that are based solely on the retrieved context, but it may also result in incomplete answers if the context doesn't contain all the necessary information. On the other hand, allowing the model to use its general knowledge can lead to more complete and helpful answers, but it also increases the risk of hallucination and providing inaccurate information. Finding the right balance between these two approaches is important for optimizing both accuracy and usefulness in RAG systems.


## Exercise 11
1. Can the model successfully combine information from multiple chunks?
- Yes, the model can successfully combine information from multiple chunks, especially when the chunks are relevant and provide complementary information. However, this can also lead to confusion if the chunks contain contradictory information or if the model struggles to piece together the information in a coherent way. The ability to combine information from multiple chunks is one of the strengths of RAG systems, but it also needs careful management of the retrieved context to ensure that it is relevant and not overwhelming for the model.

2. Does it miss information that wasn't retrieved?
- If it doesn't have retreived information, then it will just rely on its general knowledge, which may not be accurate. 

3. Does contradictory information in different chunks cause problems?
- Sometimes, yes. If the retrieved chunks contain contradictory information, it can confuse the model and lead to less accurate answers. The model may struggle to determine which information is correct or how to reconcile the contradictions, which can result in a less coherent response. This highlights the importance of ensuring that the retrieved context is not only relevant but also consistent to provide the best possible answers.








