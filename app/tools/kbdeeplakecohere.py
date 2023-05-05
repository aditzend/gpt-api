import os

from dotenv import load_dotenv
from langchain.document_loaders import (
    WebBaseLoader,
    TextLoader,
)
from langchain.indexes import VectorstoreIndexCreator
from langchain.docstore.document import Document
from langchain.embeddings import CohereEmbeddings
import requests
from langchain.document_loaders import UnstructuredWordDocumentLoader
import uuid
from langchain.llms import Cohere, OpenAI
from langchain.chat_models import ChatAnthropic
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain.vectorstores import DeepLake
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()


kb_temperature = os.getenv("KB_TEMPERATURE")
dataset_path = "/Users/alexander/clients/yzn/gpt-api/app/db"
embeddings = CohereEmbeddings()


async def download_docx_file_and_ingest(resource):
    url = resource["url"]
    local_path = f"./{str(uuid.uuid4())}.docx"
    response = requests.get(url)
    with open(local_path, "wb") as f:
        f.write(response.content)
    print(f"File downloaded successfully to {local_path} !")
    return UnstructuredWordDocumentLoader(local_path)


async def deeplake_retrieval_ingest(resources):
    print(
        f"COHERE DEEPLAKE INGESTING: {resources[0]} | Ingesting knowledge"
        " base..."
    )
    loader = await download_docx_file_and_ingest(resources[0])
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    pages = text_splitter.split_documents(documents)
    db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
    # db.add_documents(pages)


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

    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(
        llm=ChatAnthropic(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        verbose=True,
        chain_type_kwargs=chain_type_kwargs,
    )
    ans = qa({"query": query})
    print(ans)
    return ans


async def route_resource_to_loader(resource):
    if resource["type"] == "html":
        return WebBaseLoader(resource["url"]).load()
    if resource["type"] == "docx":
        response = await download_docx_file_and_ingest(resource)
        return response
    if resource["type"] == "txt":
        return TextLoader(resource["url"]).load()
