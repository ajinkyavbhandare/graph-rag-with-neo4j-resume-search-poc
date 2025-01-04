# graph-rag-with-neo4j-resume-search-poc

## Project Overview
 Graph RAG Resume Search - A proof-of-concept integrating Neo4j graph database and LangChain to enable intelligent resume search through graph relationships and retrieval-augmented generation. Built with Cypher query engine.

# Table of Contents
- [Project Overview](#project-overview)
- [Table of Contents](#table-of-contents)
- [Architecture](#architecture)
- [Installation and Setup](#installation-and-setup)
- [Improvements](#improvements)
- [Acknowledgements](#acknowledgements)

# Architecture


# Installation and Setup

1. Navigate to the directory containing source code.
2. Install the required packages using pip:

    ```bash
    $ pip install -r requirements.txt
    ```
3. Run the Streamlit app:

    ```bash
    $ chainlit run main.py
    ```

    The Streamlit app is now running and can be accessed at http://localhost:5000 in your web browser.

    ![streamlit interface](public/neo4j.png)

# Improvements
- In the cypher generation prompt, adding more few-shot examples can enhance the accuracy of cypher query generation, particularly for lower-end models.
- Similarly, modifying the question and answering prompt to elicit more concise answers can improve the accuracy of the responses.
  
# Acknowledgements
1. [None]()
