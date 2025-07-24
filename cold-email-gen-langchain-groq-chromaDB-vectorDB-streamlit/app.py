import streamlit as st
import os
import uuid
import re
import pandas as pd

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import WebBaseLoader
import chromadb

# --- Load env variables
load_dotenv()
api_key = os.getenv("GROCKEY")

# --- Load LLM
llm = ChatGroq(
    model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
    temperature=0,
    groq_api_key=api_key
)

# --- Load CSV Portfolio to Vector Store (once)
chroma_client = chromadb.PersistentClient("vectorstore")
collection = chroma_client.get_or_create_collection(name="portfolio")

df = pd.read_csv("my_portfolio.csv")
if not collection.count():
    for _, row in df.iterrows():
        collection.add(
            documents=[row["Techstack"]],
            metadatas={"links": row["Links"]},
            ids=[str(uuid.uuid4())]
        )

# --- Streamlit UI
st.title("FlickO AI Cold Email Generator")
url = st.text_input("Enter job listing URL")
submit = st.button("Generate Email")

if submit and url:
    with st.spinner("Scraping and generating email..."):
        # Load and clean job page
        loader = WebBaseLoader(url)
        page_data = loader.load().pop().page_content
        cleaned_text = re.sub(r'\s+', ' ', page_data).strip()

        # Prompt to extract job info
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the 
            following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):    
            """
        )
        chain_extract = prompt_extract | llm
        res = chain_extract.invoke({'page_data': cleaned_text})

        # Parse extracted JSON
        json_parser = JsonOutputParser()
        try:
            job = json_parser.parse(res.content)
        except:
            st.error("Could not parse job description. Please try a different link.")
            st.stop()

        # Find relevant portfolio links
        skills = job.get('skills') if isinstance(job.get('skills'), list) else [job.get('skills')]
        results = collection.query(query_texts=skills, n_results=2)
        links = [meta['links'] for meta in results.get("metadatas", []) if "links" in meta]

        # Create cold email
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Deepak, a business development executive at FlickO. FlickO is an AI & Software Consulting company dedicated to facilitating
            the seamless integration of business processes through automated tools. 
            Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
            process optimization, cost reduction, and heightened overall efficiency. 
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of FlickO 
            in fulfilling their needs.
            Also add the most relevant ones from the following links to showcase FlickO's portfolio: {link_list}
            Remember you are Deepak, BDE at FlickO. 
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):
            """
        )
        chain_email = prompt_email | llm
        email = chain_email.invoke({
            "job_description": str(job),
            "link_list": links
        }).content

        # Display result
        st.subheader("📧 Generated Cold Email:")
        st.code(email, language="markdown")
