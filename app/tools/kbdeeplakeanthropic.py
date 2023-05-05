import os

from dotenv import load_dotenv
from langchain.document_loaders import (
    WebBaseLoader,
    TextLoader,
)
from langchain.indexes import VectorstoreIndexCreator
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
import requests
from langchain.document_loaders import UnstructuredWordDocumentLoader
import uuid
from langchain.llms import OpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain.vectorstores import DeepLake
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatAnthropic


load_dotenv()


kb_temperature = os.getenv("KB_TEMPERATURE")

dataset_path = "/Users/alexander/clients/yzn/gpt-api/app/tools/edenordb"
embeddings = OpenAIEmbeddings()

local_file = "/Users/alexander/clients/yzn/gpt-api/app/edenor.txt"


# ingesting
# loader = TextLoader(local_file)
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# pages = text_splitter.split_documents(documents)
# db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
# db.add_documents(pages)

# querying
# db = DeepLake(
#     dataset_path=dataset_path, embedding_function=embeddings, read_only=True
# )


# query = "hay clase los viernes?"
# retriever = db.as_retriever()
# retriever.search_kwargs["distance_metric"] = "cos"
# retriever.search_kwargs["k"] = 4

# qa = RetrievalQA.from_chain_type(
#     llm=OpenAI(),
#     chain_type="stuff",
#     retriever=retriever,
#     return_source_documents=False,
# )


# ans = qa({"query": query})

# print(ans)


async def download_docx_file_and_ingest(resource):
    url = resource["url"]
    local_path = f"./{str(uuid.uuid4())}.docx"

    response = requests.get(url)

    with open(local_path, "wb") as f:
        f.write(response.content)

    print(f"File downloaded successfully to {local_path} !")
    return UnstructuredWordDocumentLoader(local_path)


async def deeplake_retrieval_ingest(resources):
    print(f"DEEPLAKE INGESTING: {local_file} | Ingesting knowledge base...")
    loader = TextLoader(local_file)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    pages = text_splitter.split_documents(documents)
    db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
    db.add_documents(pages)


async def deeplake_ask_retrieval(user_input, session_id, personality):
    print(f"DEEPLAKE QUESTION: {user_input} | Querying db ...")
    db = DeepLake(
        dataset_path=dataset_path,
        embedding_function=embeddings,
        read_only=True,
    )
    query = user_input
    retriever = db.as_retriever()
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["k"] = 4

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        verbose=True,
    )
    ans = qa({"query": query})
    print(ans)
    return ans


async def route_resource_to_loader(resource):
    if resource["type"] == "html":
        return WebBaseLoader(resource["url"])
    if resource["type"] == "docx":
        response = await download_docx_file_and_ingest(resource)
        return response
    if resource["type"] == "txt":
        return TextLoader(resource["url"])
