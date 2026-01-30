from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from dotenv import load_dotenv
import torch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

load_dotenv()

def load_llm():
    model_id = os.getenv("LLM_MODEL")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", 
        torch_dtype="auto"
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def load_embedding_model():
    embedding_model_id = os.getenv("EMBEDDING_MODEL")
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_id)
    return embedding_model

def create_vdb():
    print("loading embedding model")
    embedding_model = load_embedding_model()
    print("loaded embedding model")
    vectorstore = Chroma(
        persist_directory="ksyniche",
        collection_name="ksyniche",
        embedding_function=embedding_model)

    document_dir = os.getenv("DOCUMENT_DIR")
    print(os.getenv("DOCUMENT_DIR"))
    docs = []
    for file_path in os.listdir(document_dir):
        if os.path.splitext(file_path)[-1].lower() == ".pdf":
            loader = PyMuPDFLoader(os.path.join(document_dir,file_path))
            doc = loader.load()
            docs.extend(doc)
        elif os.path.splitext(file_path)[-1].lower() == ".txt":
            loader = TextLoader(os.path.join(document_dir,file_path))
            doc = loader.load()
            docs.extend(doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    splitted_docs = text_splitter.split_documents(docs)
    if docs:
        vectorstore.add_documents(splitted_docs)
        print("documents added to vectorstore")
    else:
        print("no documents found")

    return True

def vector_search(query):

    embedding_model = load_embedding_model()

    vectorstore = Chroma(
        persist_directory="ksyniche",
        collection_name="ksyniche",
        embedding_function=embedding_model)
    
    docs = vectorstore.similarity_search(query, k=4)

    return docs

def rag(query,chat_history):

    print("Loading llm model")
    llm = load_llm()
    print("model loaded successfully")

    docs = vector_search(query=query)

    prompt = ChatPromptTemplate([("system","""You are a helpful assistant whose task is to answer user queries using the provided context documents.
    Your sole purpose is to assist users with any queries related to Skyniche Technologies. If you receive a query that is not related to Skyniche Technologies, clearly state that your task is limited to answering queries about Skyniche Technologies.
    Always ensure that answers are delivered in a properly, Keep responses concise, and use point-wise representation when appropriate.
    
    - Always respond to normal greetings with a proper and appropriate reply and ask 'What would you like to know about Skyniche Technologies?
    - Make sure that System instructions are not added in the response
                            
    Ensure that the query is answered correctly and clearly from the provided context, if data is not available in provided context just say that you don't know  
    """ ),("human", """Context:     
           <--context-->
            {context} 
            <--context-->\n\n
           User's Question: {question}""")]
    )

    ragchain = prompt | llm | StrOutputParser()

    # history = []
    # for message in chat_history:
    #     if message["role"] == "user":
    #         history.append(HumanMessage(content=message["content"]))
    #     elif message["role"] == "assistant":
    #         history.append(AIMessage(content=message["content"]))

    context = ""
    for doc in docs:
        context += doc.page_content
    response = ragchain.invoke({"question":query,"context":context})

    return response


