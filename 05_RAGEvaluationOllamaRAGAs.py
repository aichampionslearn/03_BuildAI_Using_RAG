#
# Internal Notes: Balaji - 19Dec
# This code will show streaming out from a LLM. This is not RAG not does it include the Sidebar part.
# 
# Please install the following libraries : 
# !pip install langchain-ollama lancedb ragas

import ollama
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage

st.set_page_config(layout="wide", page_title="Ollama Chat")
st.title("RAG Evaluation using RAGAS")

# ====== Step 1 : Basic LLM Setup ===========

OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL = "phi3" # "llama3.1"
TEST_MESSAGE = [
    (
        "system",
        "You are a helpful assistant that answers questions with brevity"
    ),
    ("human", "Capital of Canada?"),
]

def fnAskOllama(model_name: str, messages: list):
    llm = ChatOllama(
        model=model_name,
        temperature=0.5,
        stream=False
    )
    # response = llm.invoke(messages)
    return llm #response.content
# print(fnAskOllama(LLM_MODEL,TEST_MESSAGE))

# ====== Step 2 : Load the Document and split into Chunks 

def fnGetDocument(url, filename): 
    import requests 
    url = "https://raw.githubusercontent.com/hwchase17/chroma-langchain/master/state_of_the_union.txt"
    result = requests.get(url)
    filename = "state_of_the_union.txt"

    with open(filename, "w") as f:
        f.write(result.text)

    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import CharacterTextSplitter as splitter

    document = TextLoader(filename).load()
    textSplitter = splitter(chunk_size=200, chunk_overlap=10)
    chunks = textSplitter.split_documents(documents=document)
    return chunks

url = "https://raw.githubusercontent.com/hwchase17/chroma-langchain/master/state_of_the_union.txt"
filename = "state_of_the_union.txt"
chunks = fnGetDocument(url, filename)
print(chunks[1].page_content)

# ====== Step 2 : Store in Vector DB ===========
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import LanceDB
import lancedb

embed_model = OllamaEmbeddings(model = "all-minilm")
db = lancedb.connect("/tmp/lancedb")
table = db.create_table(
    "raga_eval",
    data=[{"vector": embed_model.embed_query("Hello World"), "text": "Hello World"}],
    mode="overwrite",
)

vectorstore = LanceDB.from_documents(
    connection=db, documents=chunks, embedding=embed_model
)
retriever = vectorstore.as_retriever()

# ====== Step 3 : Setup Prompt Templates
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

llm = fnAskOllama(LLM_MODEL,TEST_MESSAGE)

template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use two sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Setup Sample Questions for Eval
from datasets import Dataset

questions = [
    "What did the president say about Justice Breyer?",
    "What did the president say about Intel's CEO?",
    "What did the president say about gun violence?",
]
ground_truth = [
    "The president said that Justice Breyer has dedicated his life to serve the country and thanked him for his service.",
    "The president said that Pat Gelsinger is ready to increase Intel's investment to $100 billion.",
    "The president asked Congress to pass proven measures to reduce gun violence.",
]
answers = []
contexts = []

# Inference
for question in questions:
    answers.append(rag_chain.invoke(question))
    contexts.append(
        [docs.page_content for docs in retriever.invoke(question)]
    )


print(answers)
dataset = {
    "question" : questions,
    "answer" : answers,
    "contexts": contexts,
    "ground_truth" : ground_truth
}

dataDictionary = Dataset.from_dict(dataset) # Convert dataset to dictionary
print(dataDictionary)

from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    faithfulness,
    answer_similarity,
    context_precision,
    context_recall,
    answer_relevancy,
    context_entity_recall,
)

# Evaluate Dataset - which is a dictionary of four components - for the listed metrics
result = evaluate(
    dataset=dataDictionary,
    metrics=[
        answer_correctness,
        faithfulness,
        answer_similarity,
        context_precision,
        context_recall,
        answer_relevancy,
        context_entity_recall,
    ],
)

df = result.to_pandas()
st.dataframe(df)
# print(df)

