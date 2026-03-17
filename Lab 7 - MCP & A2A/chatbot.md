(base) nathanctodd@NathanTodd Lab 7 - MCP & A2A % python chatbot.py 
Fetching tool schemas from Asta MCP server...
Loaded 8 tools: ['get_paper', 'get_paper_batch', 'get_citations', 'search_authors_by_name', 'get_author_papers', 'search_papers_by_relevance', 'search_paper_by_title', 'snippet_search']

Asta Research Chatbot — type 'quit' to exit

You: Hi there!
Assistant: Hello! How can I assist you today?

You: Can you tell me about the paper, "All you Need is Attention?"?
  [tool] search_paper_by_title({"title": "All you Need is Attention?", "fields": "url,year,authors"})
Assistant: The paper titled "All you Need is Attention?" was published in 2024. The authors of the paper are Junehyung Kim and Sungjae Hwang. You can find more details about the paper [here](https://www.semanticscholar.org/paper/bd6ab7b698c3258156c663e1447719891a15a540).

You: Can you tell me who all the authors of the paper are?
Assistant: Please provide the title or identifier (like DOI or Semantic Scholar ID) of the paper you're interested in, so I can find the authors for you.

You: Yes, "Attention is All you Need"
  [tool] search_paper_by_title({"title": "Attention is All you Need", "fields": "url,year,authors"})
Assistant: The paper titled **"Attention is All You Need"** was published in **2017**. It was authored by:

- Ashish Vaswani
- Noam Shazeer
- Niki Parmar
- Jakob Uszkoreit
- Llion Jones
- Aidan N. Gomez
- Lukasz Kaiser
- I. Polosukhin

You can access the paper [here](https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776).

You: Tell me more about the author Noam Shazeer
  [tool] search_authors_by_name({"name": "Noam Shazeer", "fields": "name,url,papers", "limit": 1})
Assistant: Noam M. Shazeer is an author active in the field of research, and you can find more about him on his Semantic Scholar profile [here](https://www.semanticscholar.org/author/2065683694). 

Some of his notable papers include:

1. **MÉLANGE DE RÉSEAUX NEURONAUX EXPERTS**
   - Paper ID: 3d52bf463102d1c2cf713c051401d80adadaddfa

2. **Utilisation de concepts pour le ciblage de publicités**
   - Paper ID: 272e2bf46d3fcf96ca71d161f0496df206b6a73c

If you are interested in specific publications or topics that he has worked on, please let me know!

You: Can you tell me more about paper with ID: 3d52bf463102d1c2cf713c051401d80adadaddfa
  [tool] get_paper({"paper_id": "3d52bf463102d1c2cf713c051401d80adadaddfa", "fields": "title,year,authors,abstract,journal,publicationDate,url"})