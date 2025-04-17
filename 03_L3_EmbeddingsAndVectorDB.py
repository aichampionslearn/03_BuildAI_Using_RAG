# AIChampionsHub Course 
# Course : Generative AI Mastery
# Module : Retrieval Augmented Generation (RAG)
# Lesson : 03- Deep-dive on Embedding the document chunks

# This is based on LangChain Framework and examples leverage few elements from Langchain documentation for learning purpose. 
# Objective : In this document students will be able to learn about Embeddings.
#             Embeddings are vector (i.e. an array of numbers in simple terms) representation of a piece of text. 
#             This vector is essentially used to capture the semantic meaning of the text. 
#             Mathematical operations can be performed on such a vector enabling e.g. search for text with similar in meaning. 
#             Such a search is important for retrival of information contextual to the Query or prompt by the user. 

# Instructions for Running the Code for Learning
# Step 1 : Similar to loader code is setup as set of function so that Students can just run individually to learn & Experiement. 
# Step 2 : Unless a function is called it is not going to be executed. So student can call each function one after the other.
# Step 3 : Students are encourage to change the elements like type of Embeddings to play with and learn.

# All Credit to Langchain

import warnings, os, openai, sys
import numpy as np
warnings.filterwarnings('ignore')
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings # Note this latest method as of 2024. Others are being deprecated. Refer to documentation

sys.path.append('../..')
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key  = os.environ['OPENAI_API_KEY']

# ------- Embeddings ----------
# Embedding Models take a piece of text and create a numerical representation of it.
# The base Embeddings class in LangChain provides two methods: one for embedding documents and one for embedding a query
#       .embed_documents method takes multiple texts (list)
#       .embed_query method takes single text

embed_model = OpenAIEmbeddings()
# queries = ["Ferraris are generally red", "Lamborgini can be seen in yellow", "AIChampions Hub is a great Community!"]
queries = ["King", "Queen", "Lion","Lioness"]
queries_embed = []
for q in queries:
    queries_embed.append(embed_model.embed_query(q))

for q in zip(queries, queries_embed):
    print("\n Query : ", list(q)[0], "\n Length of Query : ", len(list(q)[0]), "\n Length of Embeddings : ",len(list(q)[1]) ) # Note Embeddings vector size will be same for both queries.

# Note - Lion and Lioness have similar relationships like King and Queen. So they map to similar area in Vector Space
print("Q1 time Q2 ", np.dot(queries_embed[0], queries_embed[1])) # dot product is like matrix multiplication
print("Q3 time Q4 ", np.dot(queries_embed[2], queries_embed[3]))  

# Note - But Relationship between may be different
print("Q1 time Q3 ", np.dot(queries_embed[0], queries_embed[2]))  
print("Q2 time Q4 ", np.dot(queries_embed[1], queries_embed[3]))  

# Vector Embeddings are key to LLMs - help with Semantic meaning of words that have similar relationships or co-occurences
print("Observe the similaries between first two sentences")