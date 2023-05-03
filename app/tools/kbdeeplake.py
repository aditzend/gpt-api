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

# from chromadb import chromadb

# chroma_client = chromadb.Client(
#     Settings(
#         chroma_api_impl="rest",
#         chroma_server_host="localhost",
#         chroma_server_http_port="8000",
#     )
# )

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


async def route_resource_to_loader(resource):
    if resource["type"] == "html":
        return WebBaseLoader(resource["url"])
    if resource["type"] == "docx":
        response = await download_docx_file_and_ingest(resource)
        return response
    if resource["type"] == "txt":
        return TextLoader(resource["url"])
