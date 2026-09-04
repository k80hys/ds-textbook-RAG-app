# This script creates a data science expert using the "Introduction to Statistical Learning with Python" PDF. 
# It uses Langchain to load the PDF, split it into chunks, create embeddings, 
# and set up a retrieval-augmented generation (RAG) chain with an LLM model to answer questions based on the content of the PDF.


# To run this program on a different information source: 
# 1. Replace the PDF in /pdf and update the path in the PyPDFLoader
# 2. Delete the existing vector database in /data/chroma_statistical_learning.db
# 3. Re-run the script to create a new vector database from the updated PDF
# 4. Update the questions in the RAG chain to ask about the new information source
# 5. Optional: Adjust chunk size in the CharacterTextSplitter for performance and context length

# Packages and their roles:
# Langchain PDF loader
from langchain_community.document_loaders import PyPDFLoader
# Langchain Text Splitter - splits text into chunks for embedding
from langchain_text_splitters import CharacterTextSplitter
# Langchain Vector Store - creates vector db to store embeddings for retrieval
from langchain_community.vectorstores import Chroma
# Langchain LLM Model
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# Langchain RAG Chain - enables retrieval-augmented generation
from langchain_core.runnables import RunnablePassthrough
# Langchain Output Parser - parses output from LLM
from langchain_core.output_parsers import StrOutputParser
# Langchain Prompt Template - creates prompt for LLM
from langchain_core.prompts import ChatPromptTemplate

import pandas as pd
import yaml
from pprint import pprint

# Insert your OpenAI API key in a credentials.yml file in the same directory as this script
OPENAI_API_KEY = yaml.safe_load(open('credentials.yml'))['openai']
LLM_MODEL = "gpt-5.4-mini"

## Load document and split into chunks for embedding

# PDF Loader 
loader = PyPDFLoader("pdf/ISLP_website.pdf")

# This takes a few minutes to run if the PDF is large
# In this project it's ~600 pages, so it takes a few minutes to load and split into chunks
documents = loader.load()

# Chunk size for this project is 1000 to balance between context length and performance
CHUNK_SIZE = 1000

# Create a text splitter to split the documents into chunks
text_splitter = CharacterTextSplitter(
    chunk_size=CHUNK_SIZE, 
    # chunk_overlap=100,
    separator="\n"
)

# Split the documents into chunks for embedding
docs = text_splitter.split_documents(documents)

docs

len(docs)

docs[0]

# Shows the content of the 6th document chunk
pprint(dict(docs[5])["page_content"])

docs[5]

## Create Vector Database

# Create the embedding function using OpenAI's text-embedding-ada-002 model
embedding_function = OpenAIEmbeddings(
    model='text-embedding-ada-002',
    api_key=OPENAI_API_KEY
)

# Create the vector store using Chroma to store the embeddings for retrieval
vectorstore = Chroma.from_documents(
    docs, 
    persist_directory="data/chroma_statistical_learning.db",
    embedding=embedding_function
)

# Persist the vector store for future use in the app
vectorstore = Chroma(
    persist_directory="data/chroma_statistical_learning.db",
    embedding_function=embedding_function
)

# Create a retriever from the vector store to retrieve relevant documents based on user queries
retriever = vectorstore.as_retriever()

retriever

## RAG LLM Model

# Create a prompt template for the RAG chain that includes the context and the user question
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

# Create a ChatPromptTemplate from the template string
prompt = ChatPromptTemplate.from_template(template)

# Create the ChatOpenAI model for the RAG chain
# Reasoning set to "none" to avoid unnecessary steps
model = ChatOpenAI(
    model=LLM_MODEL,
    use_responses_api=True,
    reasoning={"effort": "none"},
    api_key=OPENAI_API_KEY
)

# Create the RAG chain by combining the retriever, prompt, and model
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Invoke the RAG chain with a test question about PCA and print the result
result = rag_chain.invoke("What are the top 3 things needed to do principal component analysis (pca)?")

pprint(result)

# Invoke the RAG chain with a test question about linear regression and print the result
result = rag_chain.invoke("What are the different types and variants of linear regression, and how do they differ?")

pprint(result)
