# AIChampionsHub Course 
# Course : Generative AI Mastery
# Module : Retrieval Augmented Generation (RAG)
# Lesson : 01 - Deep-dive on Document Loaders in Langchain

# This is based on LangChain Framework and examples leverage few elements from Langchain documentation for learning purpose. 
# Objective : In this document students will be able to read content from PDF, Multiple PDFs, from video and from Website. 

# Instructions for Running the Code for Learning
# Step 1 : Each type of Loader is setup as a function so that Students can just run that particular type of Loader to learn & Experiement. 
# Step 2 : Unless a function is called it is not going to be executed. So student can call each function one after the other.
# Step 3 : Students are encourage to change the elements like document or directory to play with and learn.


import os, sys, warnings 
warnings.filterwarnings('ignore')
sys.path.append('../..')

from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

def fn_loadllm():
    MODEL="llama3.1"
    llm = ChatOllama(model=MODEL,temperature=0)
    return llm;

def fn_testllm(llm, prompt):
    # from langchain_core.messages import HumanMessage, AIMessage
    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to Hindi. Translate the user sentence.",
        ),
        ("human", prompt),
    ]
    response = llm.invoke(messages)
    print(response.content)
    return response;

#-------  Load contents from one PDF File -------
def fnLoadPDFData(doc): 
    #------- Load a PDF Using Document Loader ----------
    loader = PyPDFLoader(doc)
    pages = loader.load() # Returns a list

    PAGE_NUMBER  = 0
    page = pages[PAGE_NUMBER]
    print("\n------ PDF Content fnLoadPDFData ---------\n")
    print("\n\n Metadata \n", pages[PAGE_NUMBER].metadata)  # This is at a Individual Page level
    print("\n\nPage Content \n", pages[PAGE_NUMBER].page_content[0:300]) # This is at a Individual Page level
    return; 

#-------  Load contents from set of PDF files in a Directory -------
def fnLoadPDFFromDirectoryMethod01():
    print("\n------ PDF Content fnLoadPDFFromDirectory Selected Load ---------\n")

    # Note: Sometimes we may not be sure of what is in a directory and so want to load specific files    
    loaders = [
        PyPDFLoader("pdfs/MachineLearning-Lecture01.pdf"),
        PyPDFLoader("pdfs/language_models_are_few_shot_learners.pdf")
        ]
    docs = []
    for loader in loaders:
        print("\n")
        pages = loader.load()
        docs.extend(pages)
    return docs;

def fnLoadPDFFromDirectoryMethod02():
    #from langchain_community.document_loaders import DirectoryLoader
    from langchain_community.document_loaders import PyPDFDirectoryLoader

  
    print("\n------ PDF Content fnLoadPDFFromDirectory Bulk Load---------\n")
    loader = PyPDFDirectoryLoader(
        path = "./pdfs/",
        glob = "**/[!.]*.pdf",
        silent_errors = False,
        load_hidden = False,
        recursive = False,
        extract_images = False,
        password = None,
        mode = "page"
    )
    
    docs = loader.load()
    return docs;

#-------  Load a Video Content from Youtube. -------
def fnLoadVideoData(): 
    # ! pip install yt_dlp
    # ! pip install pydub
    print("\n --- Load Video Content ---- \n\n")
    
    from langchain_community.document_loaders.generic import GenericLoader, FileSystemBlobLoader
    from langchain_community.document_loaders.parsers import OpenAIWhisperParser
    from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

    url="https://www.youtube.com/watch?v=jGwO_UgTS7I" # Link to CS229 ML Video by Professor Andrew Ng
    # https://youtube.com/shorts/vpB-tlUB4HA?si=vW9ABeBPkMe_vVK7
    # url="https://youtube.com/shorts/W_RT_6vquCY?si=mcy9KeKI5SJKWknH"
    save_dir="./videos/"
    loader = GenericLoader(
        YoutubeAudioLoader([url],save_dir),  # fetch from youtube.  Note First time download from Web. then use local file
        #FileSystemBlobLoader(save_dir, glob="*.m4a"),   #fetch locally
        OpenAIWhisperParser()
    )
    docs = loader.load()
    # print(docs)
    print("Video Content \n", docs[0].page_content[0:200])
    return

#-------  Load content from a Website. -------
def fnLoadWebData(): 
   
    print("\n --- Load a Web content ---- \n\n")
    
    from langchain_community.document_loaders import WebBaseLoader
    loader = WebBaseLoader("https://github.com/basecamp/handbook/blob/master/titles-for-programmers.md")
    docs = loader.load()
    print("\n Metadata \n", docs[0].metadata)
    print("\n Web Content \n", docs[0].page_content[0:200])

    return;

#=========  Placeholder for Execution ========
if __name__ == "__main__": 
    
    llm = fn_loadllm()
    fn_testllm(llm,"I love Artificial Intelligence")
    
    fnLoadPDFData("pdfs/MachineLearning-Lecture01.pdf")
    
    docs = fnLoadPDFFromDirectoryMethod01()
    print("\nTotal number of pages of all the documents : ", len(docs))
    print(docs[0].page_content[0:300])

    docs = fnLoadPDFFromDirectoryMethod02()
    if len(docs):
       print("\nTotal number of pages of last document : ", len(docs))
       print(docs[0].page_content[0:300])
        
    # fnLoadVideoData()
    fnLoadWebData()
    
    