from dotenv import load_dotenv
from langchain.document_loaders import (
    DirectoryLoader,
    WebBaseLoader,
    TextLoader,
)
from langchain.indexes import VectorstoreIndexCreator
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA, ChatVectorDBChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import requests
from langchain.document_loaders import UnstructuredWordDocumentLoader
import uuid
from langchain.llms import OpenAI
import redis
import os
from langchain.schema import AIMessage, HumanMessage, SystemMessage


load_dotenv()

redis_host = os.getenv("REDIS_HOST") or "localhost"
redis_port = os.getenv("REDIS_PORT") or 6379

docsearch = ""
qa = ""
index = ""


def connect_to_redis():
    return redis.Redis(host=redis_host, port=redis_port, db=0)


def add_message_to_chat_history(session_id, user_message, response):
    redis_conn = connect_to_redis()
    user_key = f"user:{session_id}"
    user_value = user_message
    redis_conn.set(user_key, str(user_value))
    bot_key = f"bot:{session_id}"
    bot_value = response
    redis_conn.set(bot_key, str(bot_value))


def get_chat_history(session_id):
    redis_conn = connect_to_redis()
    user_key = f"user:{session_id}"
    if redis_conn.get(user_key) is None:
        return []
    user_message = redis_conn.get(user_key).decode()
    bot_key = f"bot:{session_id}"
    bot_response = redis_conn.get(bot_key).decode()
    return [(user_message, bot_response)]


async def download_docx_file_and_ingest(resource):
    url = resource["url"]
    local_path = f"./{str(uuid.uuid4())}.docx"

    response = requests.get(url)

    with open(local_path, "wb") as f:
        f.write(response.content)

    print(f"File downloaded successfully to {local_path} !")
    return UnstructuredWordDocumentLoader(local_path)


def kb_qa(query):
    print(f"QUESTION: {query} | Querying knowledge base...")
    loader = DirectoryLoader("./kb")
    index = VectorstoreIndexCreator().from_loaders([loader])
    result = index.query_with_sources(query)
    print(result)
    return result


async def route_resource_to_loader(resource):
    if resource["type"] == "HTML":
        return WebBaseLoader(resource["url"])
    if resource["type"] == "DOCX":
        response = await download_docx_file_and_ingest(resource)
        return response


async def ingest(resources):
    print(f"INGESTING: {resources} | Ingesting knowledge base...")
    loaders = [
        await route_resource_to_loader(resource) for resource in resources
    ]
    global index
    print(loaders[0])
    index = VectorstoreIndexCreator().from_loaders(loaders)
    return {"status": "READY_TO_ANSWER"}


async def conversational_ingest(resources):
    print(
        f"INGESTING: {resources} | Ingesting conversational knowledge base..."
    )
    documents = []
    for resource in resources:
        loader = await route_resource_to_loader(resource)
        documents += loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()

    global docsearch
    docsearch = Chroma.from_documents(texts, embeddings)
    global qa
    qa = ChatVectorDBChain.from_llm(OpenAI(temperature=0), docsearch)
    return {"status": "READY_TO_ANSWER"}


def ask_retrieval(user_input, session_id, personality):
    print(f"QUESTION: {user_input} | Querying web...")
    global index
    result = index.query_with_sources(user_input)
    print(result)
    return result


def ask_conversational(user_input, session_id, personality):
    print(f"QUESTION: {user_input} | Querying conversational model...")
    chat_history = get_chat_history(session_id)
    result = qa({"question": user_input, "chat_history": chat_history})
    add_message_to_chat_history(
        session_id, user_message=user_input, response=result["answer"]
    )
    print(result)
    return result
