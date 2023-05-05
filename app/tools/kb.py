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

from chromadb.config import Settings


load_dotenv()

redis_host = os.getenv("REDIS_HOST") or "localhost"
redis_port = os.getenv("REDIS_PORT") or 6379

allow_index_deletion = os.getenv("ALLOW_INDEX_DELETION")
kb_temperature = os.getenv("KB_TEMPERATURE")

index = ""
docsearch = ""
persist_directory = "/externaldb"
embedding = OpenAIEmbeddings()
KB = Chroma(
    collection_name="external",
    persist_directory=persist_directory,
    client_settings=Settings(
        chroma_api_impl="rest",
        chroma_server_host="localhost",
        chroma_server_http_port="8000",
    ),
)


async def download_docx_file_and_ingest(resource):
    url = resource["url"]
    local_path = f"./{str(uuid.uuid4())}.docx"

    response = requests.get(url)

    with open(local_path, "wb") as f:
        f.write(response.content)

    print(f"File downloaded successfully to {local_path} !")
    return UnstructuredWordDocumentLoader(local_path)


async def v3_retrieval_ingest(resources):
    print(f"V3 INGESTING: {resources} | Ingesting knowledge base...")
    loaders = [
        await route_resource_to_loader(resource) for resource in resources
    ]
    global index
    index = VectorstoreIndexCreator().from_loaders(loaders)
    print(index)


async def v3_ask_retrieval(user_input, session_id, personality):
    print(f"QUESTION: {user_input} | Querying web...")
    global index
    result = index.query_with_sources(user_input)
    print(result)
    return result


def kb_qa(query):
    print(f"QUESTION: {query} | Querying knowledge base...")
    loader = DirectoryLoader("./kb")
    index = VectorstoreIndexCreator().from_loaders([loader])
    result = index.query_with_sources(query)
    print(result)
    return result


async def route_resource_to_loader(resource):
    if resource["type"] == "html":
        return WebBaseLoader(resource["url"])
    if resource["type"] == "docx":
        response = await download_docx_file_and_ingest(resource)
        return response
    if resource["type"] == "txt":
        return TextLoader(resource["url"])


async def retrieval_ingest(resources):
    print(f"INGESTING: {resources} | Ingesting knowledge base...")
    loaders = [
        await route_resource_to_loader(resource) for resource in resources
    ]
    global index
    print(loaders[0])
    index = VectorstoreIndexCreator().from_loaders(loaders)
    return {"status": "READY_TO_ANSWER"}


def ask_retrieval(user_input, session_id, personality):
    print(f"QUESTION: {user_input} | Querying web...")
    global index
    result = index.query_with_sources(user_input)
    print(result)
    return result


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
    docsearch = KB.from_documents(texts, embeddings)
    print(KB)
    global index
    index = ChatVectorDBChain.from_llm(
        OpenAI(temperature=kb_temperature), docsearch
    )
    return {"status": "READY_TO_ANSWER"}


def ask_conversational(user_input, session_id, personality):
    print(f"QUESTION: {user_input} | Querying conversational model...")
    chat_history = get_chat_history(session_id)
    print(f"\n Chat history for session {session_id}: {chat_history}\n")
    result = index({"question": user_input, "chat_history": chat_history})
    add_message_to_chat_history(
        session_id, user_message=user_input, response=result["answer"]
    )
    print(result)
    return result


def make_post_prompts_from_copy(copy):
    post_example = f"""
      Ya se puede jugar con GPT-4. 🤯

      Cómo hacer para entrar?

      El preview está abierto solo para los usuarios de ChatGPT Plus, una subscripción que cuesta 20 USD/mes.

      Como somos varios los que queremos probarlo cuanto antes, OpenAI puso un tope a la cantidad de mensajes que podemos enviarle: 100 mensajes cada 4 horas.

      Alguien ya lo estuvo probando?

      #tecnologia#Automation#AI#machinelearning
          """
    prompt = f"""
      Act as the best post writer in the world. Use this post as an example of your writing style: 

      {post_example}

      ###

      Write a LinkedIn post, an Instagram post and a tweet about the following article. All 3 posts must be in argentinean spanish. 

      Article:

      {copy}
      """
    return prompt


def make_post_prompts_from_url(url, post_example):
    article = WebBaseLoader(url)
    article = article.load()
    article = article[0].page_content
    prompt = f"""
      Act as the best post writer in the world. Use this post as an example of your writing style: 

      {post_example}

      ###

      Write a LinkedIn post, an Instagram post and a tweet about the following article. All 3 posts must be in argentinean spanish. 

      Article:

      {article}
      """
    return prompt


def delete_all_indexes():
    if allow_index_deletion:
        global index
        index = ""
        global docsearch
        docsearch = ""
        print("Indexes deleted")
        return "Indexes deleted"
    else:
        return "Index deletion not allowed"


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
