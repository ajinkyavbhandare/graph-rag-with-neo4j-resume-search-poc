# command to run on local network: chainlit run main.py --host 0.0.0.0 --port 8501

# Standard Library Imports
import asyncio
import json
import logging
import os
import pickle
import platform
import random
import re
import shutil
import subprocess
import time
import uuid
import winreg
from pathlib import Path
from time import sleep

# Third-Party Libraries
import chromadb
import dotenv
import google.generativeai as genai
import networkx as nx
import pandas as pd
import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from networkx.readwrite import json_graph
from selenium.common.exceptions import (
    NoSuchElementException, 
    TimeoutException, 
    WebDriverException
)
from selenium.webdriver.common.by import By

# Machine Learning and AI Libraries
from langchain_community.document_loaders import JSONLoader
from langchain_community.graphs import (
    Neo4jGraph, 
    NetworkxEntityGraph
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate, 
    PromptTemplate
)
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain

# Chainlit Related Imports
import chainlit as cl
from chainlit.input_widget import (
    Select, 
    Switch, 
    Slider
)

# Vector Database Imports
from langchain_chroma import Chroma
import chromadb

# Graph-related Imports
from langchain.chains import (
    GraphCypherQAChain, 
    GraphQAChain,
    RetrievalQA,
)


os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ['NEO4J_URL'] = os.getenv("NEO4J_URL")
os.environ['NEO4J_USERNAME'] = os.getenv("NEO4J_USERNAME")
os.environ['NEO4J_PASSWORD'] = os.getenv("NEO4J_PASSWORD")

#_____________________________________________ G R A P H   R A G   N E O 4 J    C Y P H E T Q A C H A I N _________________________
def get_neo4j_chain():
    with open('./data/json/data.json', 'r') as f:
        data = json.load(f)

    def format_json_custom(input_json):
        """
        Convert JSON to a custom string format with unquoted keys and quoted values.

        :param input_json: Input JSON data
        :return: Formatted string representation
        """
        # Convert to JSON string first
        json_str = json.dumps(input_json, indent=4)
        #json_str = str(input_json)
        # Remove quotes around keys
        json_str = re.sub(r'^(\s*)"(\w+)":', r'\1\2:', json_str, flags=re.MULTILINE)

        # Ensure string values are single-quoted
        #json_str = re.sub(r': "(.*?)"', r": '\1'", json_str)

        return json_str

    input_data = str(format_json_custom(data))

    qa_llm = ChatGoogleGenerativeAI(model= os.getenv("QA_LLM"))
    cypher_llm = ChatGoogleGenerativeAI(model=  os.getenv("CYPHER_LLM"))

    graph = Neo4jGraph(
        url=os.environ['NEO4J_URL'],
        username=os.environ['NEO4J_USERNAME'],
        password=os.environ['NEO4J_PASSWORD'],
        enhanced_schema=True
        )

    graph.sanitize = True
    graph.query("""
    MATCH (n)
    DETACH DELETE n;""")
    graph.refresh_schema()

    graph.query(
        """
    UNWIND """+ input_data + """ AS profile
    MERGE (p:Person {
        name: coalesce(profile.personal_info.name, 'not specified'),
        location: coalesce(profile.personal_info.location, 'not specified')
    })
    SET p.title = coalesce(profile.about.title, 'not specified'),
        p.goal = coalesce(profile.about.goal, 'not specified'),
        p.focus = coalesce(profile.about.focus, 'not specified'),
        p.belief = coalesce(profile.about.belief, 'not specified')

    WITH p, profile
    UNWIND (CASE WHEN profile.top_skills.skills = [] THEN ['not specified'] ELSE profile.top_skills.skills END) AS skill
    MERGE (s:Skill {name: skill})
    MERGE (p)-[:HAS_SKILL]->(s)

    WITH p, profile
    UNWIND (CASE WHEN profile.experience = [] THEN [{}] ELSE profile.experience END) AS exp
    MERGE (e:Experience {
        title: coalesce(exp.title, 'not specified'),
        company: coalesce(exp.company, 'not specified'),
        date: coalesce(exp.date, 'not specified')
    })
    WITH p, profile, e, exp
    UNWIND (CASE WHEN exp.responsibilities = [] THEN ['not specified'] ELSE exp.responsibilities END) AS responsibility
    SET e.responsibilities = coalesce(e.responsibilities, []) + responsibility
    MERGE (p)-[:HAS_EXPERIENCE]->(e)

    WITH p, profile
    UNWIND (CASE WHEN profile.education = [] THEN [{}] ELSE profile.education END) AS edu
    MERGE (ed:Education {
        degree: coalesce(edu.degree, 'not specified'),
        institution: coalesce(edu.institution, 'not specified'),
        date: coalesce(edu.date, 'not specified')
    })
    MERGE (p)-[:HAS_EDUCATION]->(ed)

    WITH p, profile
    UNWIND (CASE WHEN profile.projects = [] THEN [{}] ELSE profile.projects END) AS proj
    MERGE (pr:Project {name: coalesce(proj.name, 'not specified')})
    MERGE (p)-[:HAS_PROJECT]->(pr)

    WITH p, profile
    UNWIND (CASE WHEN profile.honors_and_awards = [] THEN [{}] ELSE profile.honors_and_awards END) AS award
    MERGE (aw:Award {
        name: coalesce(award.award, 'not specified'),
        event: coalesce(award.event, 'not specified'),
        year: coalesce(award.year, 'not specified')
    })
    MERGE (p)-[:RECEIVED_AWARD]->(aw)

    WITH p, profile
    UNWIND (CASE WHEN profile.organizations = [] THEN [{}] ELSE profile.organizations END) AS org
    MERGE (o:Organization {
        name: coalesce(org.name, 'not specified'),
        role: coalesce(org.role, 'not specified')
    })
    MERGE (p)-[:MEMBER_OF]->(o)

    WITH p, profile
    UNWIND (CASE WHEN profile.languages = [] THEN [{}] ELSE profile.languages END) AS lang
    MERGE (l:Language {
        name: coalesce(lang.language, 'not specified'),
        proficiency: coalesce(lang.proficiency, 'not specified')
    })
    MERGE (p)-[:SPEAKS]->(l)
        """
    )

    graph.refresh_schema()

    CYPHER_GENERATION_TEMPLATE ="""
    Instructions:
    Use only the provided relationship types and properties in the schema.
    Do not use any other relationship types or properties that are not provided.
    Schema:
    {schema}
    Note: Do not include any explanations or apologies in your responses.
    Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
    Do not include any text except the generated Cypher statement.
    Examples: Here are a few examples of generated Cypher statements for particular questions:
    # How many people played in Top Gun?
    MATCH (m:Movie {{name:"Top Gun"}})<-[:ACTED_IN]-()

    **Cypher Query:**
    MATCH (n) WHERE apoc.fuzzySearch(n.property, '{query}', 0.8)
    RETURN n.property
    RETURN count(*) AS numberOfActors

    Task: Generate Cypher statement to query a graph database using advanced matching techniques.
    Instructions:
    - Utilize APOC text similarity functions for flexible matching
    - Handle potential spelling variations and typos
    - Use only the provided relationship types and properties in the schema

    Schema:
    {schema}

    Matching Strategies:
    1. Levenshtein Distance Matching
    2. Jaro-Winkler Similarity Matching
    3. Combine multiple similarity techniques

    Example Matching Techniques:
    # Levenshtein Distance Matching (allows character insertions/deletions)
        MATCH (p:Person)-[:HAS_EXPERIENCE]->(e:Experience)
    WHERE
        apoc.text.levenshteinSimilarity(toLower(e.company), toLower('codeworks')) >= 0.8 OR
        apoc.text.levenshteinDistance(toLower(e.company), toLower('codeworks')) <= 2
    RETURN p

    # Jaro-Winkler Similarity Matching (better for similar strings)
    MATCH (p:Person)-[:HAS_EXPERIENCE]->(e:Experience)
    WHERE apoc.text.jaroWinklerDistance(e.company, 'CodeWorks') >= 0.7  // Lower threshold
    OR toLower(e.company) = 'codeworks'  // Check for exact match (lowercase)
    RETURN p

    # Combined Matching Strategy
    MATCH (p:Person)-[:HAS_EXPERIENCE]->(e:Experience)
    WHERE
        toLower(e.company) = toLower('CodeWorks') OR
        apoc.text.levenshteinDistance(toLower(e.company), toLower('CodeWorks')) <= 2 OR
        apoc.text.jaroWinklerDistance(toLower(e.company), toLower('CodeWorks')) >= 0.8
    RETURN p

    Notes:
    - Levenshtein Distance: Number of single-character edits required
    - Lower values mean closer matches
    - '<=2' allows up to 2 character changes
    - Jaro-Winkler Similarity:
    - Range 0-1 (1 = perfect match)
    - Better for handling transpositions
    - Adjust thresholds based on your specific matching needs

    Do not include any explanations or apologies.
    Do not respond to anything other than constructing a Cypher statement.
    Do not include any text except the generated Cypher statement.

    The question is:
    {question}
    """

    QA_TEMPLATE = """
    You are an expert at interpreting graph database query results and providing clear, concise answers.

    Context:
    - You will receive a specific question
    - You will have access to the Cypher query(which was generated from question) results
    - it will not include the cyphar query no it will not contain anything ralated to the question but it is result of question

    Instructions:
    1. Carefully analyze the context and query results
    2. Provide a direct, informative answer to the original question
    3. If no results are found, clearly state that no matching information was discovered
    4. Format the answer to be human-readable and easy to understand
    5. Only use information present in the provided context

    Here’s an updated version of the note to be included in the prompt:

    You are provided with a result that is most likely the answer to the question. Do not search for or infer additional context or information beyond what is explicitly provided in the result. Assume that the result directly answers the question, and avoid statements such as "there is no mention of [X]." Simply present the result as the answer in a clear and concise manner.
    Example:
    Question: Who speaks English?
    Result: John
    Answer: John speaks English.

    note: if you recive emty list in the answer it can be the issue in query genrtion so request user to rerun the query or it might be the result

    Question:
    {question}

    result:
    {context}

    Your detailed answer:
    """

    QA_PROMPT = PromptTemplate(
        input_variables=[ "question", "context"], template=QA_TEMPLATE,
    )

    CYPHER_PROMPT = PromptTemplate(
        input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
    )

    chain = GraphCypherQAChain.from_llm(
        graph=graph,
        cypher_llm=cypher_llm,
        qa_llm=qa_llm,
        qa_prompt=QA_PROMPT,
        cypher_prompt=CYPHER_PROMPT,
        verbose=True,
        validate_cypher=True,
        return_intermediate_steps=True,
        allow_dangerous_requests=True,
    )
    return chain


@cl.on_settings_update
async def setup_agent(settings):
    print("Selected RAG Method:", settings['Model']) 
    if settings['Model'] == "graph rag neo4j":
        await cl.Message(content="Configuring Graph RAG with Neo4j...").send()
        loop = asyncio.get_event_loop()
        chain = await loop.run_in_executor(None, get_neo4j_chain)
        cl.user_session.set("chain", chain)
        await cl.Message(content="Configured Graph RAG with Neo4j ✅").send()

@cl.on_chat_start
async def start():

    print(cl.user_session.get('major_version'))
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="RAG - Method",
                values=["none", "graph rag neo4j"],
                initial_index=0,
            )
            
        ]
    ).send()


@cl.on_message
async def main(message: str):
    chain = cl.user_session.get("chain")
    response = await chain.acall(message.content, callbacks=[cl.AsyncLangchainCallbackHandler()])
    await cl.Message(response["result"]).send()
