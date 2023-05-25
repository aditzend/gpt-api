from dotenv import load_dotenv
import os
from langchain.embeddings.openai import OpenAIEmbeddings

import time
import math

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from tools.openaidirect import (
    system_completion_v1_turbo_t0,
    system_user_v1_turbo_t0,
    system_user_v1_turbo_t0_full,
)
import logging

import uuid
import requests

import redis
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.document_loaders import (
    DirectoryLoader,
    WebBaseLoader,
    TextLoader,
    UnstructuredPDFLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    YoutubeLoader,
)

logger = logging.getLogger("uvicorn")

redis_host = os.getenv("REDIS_HOST") or "localhost"
redis_port = os.getenv("REDIS_PORT") or 6379

load_dotenv()


def ingest_resource(resource):
    if resource["type"] == "web":
        download_html_file_and_ingest(resource)
    if resource["type"] == "docx":
        download_docx_file_and_ingest(resource)
    if resource["type"] == "txt":
        download_txt_file_and_ingest(resource)
    if resource["type"] == "csv":
        download_csv_file_and_ingest(resource)
    if resource["type"] == "pptx":
        download_pptx_file_and_ingest(resource)
    if resource["type"] == "pdf":
        download_pdf_file_and_ingest(resource)
    if resource["type"] == "jpg":
        download_jpg_file_and_ingest(resource)
    if resource["type"] == "png":
        download_png_file_and_ingest(resource)
    if resource["type"] == "youtube":
        youtube_ingest(resource)


def connect_to_redis():
    return redis.Redis(host=redis_host, port=redis_port, db=0)


def check_index_existance(index="main") -> bool:
    redis_conn = connect_to_redis()
    index_key = f"indexes:{index}"
    if redis_conn.get(index_key) is None:
        return False
    else:
        return True


def youtube_ingest(resource):
    logger.info(f"[[download_youtube_file_and_ingest]] starting ...")
    url = resource["url"]
    index = resource["index"]
    index_exists: bool = check_index_existance(index)

    loader = YoutubeLoader.from_youtube_url(
        url, language="es", add_video_info=True
    )
    doc = loader.load()
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100
    )
    chunks = text_splitter.split_documents(doc)
    embeddings = OpenAIEmbeddings()

    if not index_exists:
        index_db = FAISS.from_documents(chunks, embedding=embeddings)
    else:
        index_db = FAISS.load_local(index, embeddings=embeddings)
        index_db.add_documents(chunks)

    index_db.save_local(index)
    logger.info(f"Index {index} saved locally! {index_db.docstore._dict}")


def download_png_file_and_ingest(resource):
    logger.info(f"[[download_png_file_and_ingest]] starting ...")
    url = resource["url"]
    index = resource["index"]
    index_exists: bool = check_index_existance(index)

    local_path = f"./{index}-{str(uuid.uuid1())}.png"

    response = requests.get(url)

    with open(local_path, "wb") as f:
        f.write(response.content)
    logger.info(f"File downloaded successfully to {local_path} !")

    loader = UnstructuredImageLoader(local_path)
    doc = loader.load()
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100
    )
    chunks = text_splitter.split_documents(doc)
    embeddings = OpenAIEmbeddings()

    if not index_exists:
        index_db = FAISS.from_documents(chunks, embedding=embeddings)
    else:
        index_db = FAISS.load_local(index, embeddings=embeddings)
        index_db.add_documents(chunks)

    index_db.save_local(index)
    logger.info(f"Index {index} saved locally! {index_db.docstore._dict}")


def download_jpg_file_and_ingest(resource):
    logger.info(f"[[download_jpg_file_and_ingest]] starting ...")
    url = resource["url"]
    index = resource["index"]
    index_exists: bool = check_index_existance(index)

    local_path = f"./{index}-{str(uuid.uuid1())}.jpg"

    response = requests.get(url)

    with open(local_path, "wb") as f:
        f.write(response.content)
    logger.info(f"File downloaded successfully to {local_path} !")

    loader = UnstructuredImageLoader(local_path)
    doc = loader.load()
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100
    )
    chunks = text_splitter.split_documents(doc)
    embeddings = OpenAIEmbeddings()

    if not index_exists:
        index_db = FAISS.from_documents(chunks, embedding=embeddings)
    else:
        index_db = FAISS.load_local(index, embeddings=embeddings)
        index_db.add_documents(chunks)

    index_db.save_local(index)
    logger.info(f"Index {index} saved locally! {index_db.docstore._dict}")


def download_pdf_file_and_ingest(resource):
    logger.info(f"[[download_pdf_file_and_ingest]] starting ...")
    url = resource["url"]
    index = resource["index"]
    index_exists: bool = check_index_existance(index)

    local_path = f"./{index}-{str(uuid.uuid1())}.pdf"

    response = requests.get(url)

    with open(local_path, "wb") as f:
        f.write(response.content)
    logger.info(f"File downloaded successfully to {local_path} !")

    loader = UnstructuredPDFLoader(local_path)
    doc = loader.load()
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100
    )
    chunks = text_splitter.split_documents(doc)
    embeddings = OpenAIEmbeddings()

    if not index_exists:
        index_db = FAISS.from_documents(chunks, embedding=embeddings)
    else:
        index_db = FAISS.load_local(index, embeddings=embeddings)
        index_db.add_documents(chunks)

    index_db.save_local(index)
    logger.info(f"Index {index} saved locally! {index_db.docstore._dict}")


def download_pptx_file_and_ingest(resource):
    logger.info(f"[[download_pptx_file_and_ingest]] starting ...")
    url = resource["url"]
    index = resource["index"]
    index_exists: bool = check_index_existance(index)

    local_path = f"./{index}-{str(uuid.uuid1())}.pptx"

    response = requests.get(url)

    with open(local_path, "wb") as f:
        f.write(response.content)
    logger.info(f"File downloaded successfully to {local_path} !")

    loader = UnstructuredPowerPointLoader(local_path)
    doc = loader.load()
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100
    )
    chunks = text_splitter.split_documents(doc)
    embeddings = OpenAIEmbeddings()

    if not index_exists:
        index_db = FAISS.from_documents(chunks, embedding=embeddings)
    else:
        index_db = FAISS.load_local(index, embeddings=embeddings)
        index_db.add_documents(chunks)

    index_db.save_local(index)
    logger.info(f"Index {index} saved locally! {index_db.docstore._dict}")


def download_docx_file_and_ingest(resource):
    logger.info(f"[[download_docx_file_and_ingest]] starting ...")
    url = resource["url"]
    index = resource["index"]
    index_exists: bool = check_index_existance(index)

    local_path = f"./{index}-{str(uuid.uuid1())}.docx"

    response = requests.get(url)

    with open(local_path, "wb") as f:
        f.write(response.content)
    logger.info(f"File downloaded successfully to {local_path} !")

    loader = UnstructuredWordDocumentLoader(local_path)
    doc = loader.load()
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100
    )
    chunks = text_splitter.split_documents(doc)
    embeddings = OpenAIEmbeddings()

    if not index_exists:
        index_db = FAISS.from_documents(chunks, embedding=embeddings)
    else:
        index_db = FAISS.load_local(index, embeddings=embeddings)
        index_db.add_documents(chunks)

    index_db.save_local(index)
    logger.info(f"Index {index} saved locally! {index_db.docstore._dict}")


def download_csv_file_and_ingest(resource):
    logger.info(f"[[download_csv_file_and_ingest]] starting ...")
    url = resource["url"]
    index = resource["index"]
    index_exists: bool = check_index_existance(index)

    local_path = f"./{index}-{str(uuid.uuid1())}.csv"

    response = requests.get(url)

    with open(local_path, "wb") as f:
        f.write(response.content)

    logger.info(f"File downloaded successfully to {local_path} !")

    loader = CSVLoader(
        file_path=local_path,
        csv_args={
            "delimiter": ",",
            #  "quotechar": csv.Dialect.quotechar,
        },
    )
    doc = loader.load()
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100
    )
    chunks = text_splitter.split_documents(doc)
    embeddings = OpenAIEmbeddings()

    if not index_exists:
        index_db = FAISS.from_documents(chunks, embedding=embeddings)
    else:
        index_db = FAISS.load_local(index, embeddings=embeddings)
        index_db.add_documents(chunks)

    index_db.save_local(index)
    logger.info(f"Index {index} saved locally! {index_db.docstore._dict}")


def download_txt_file_and_ingest(resource):
    logger.info(f"[[download_txt_file_and_ingest]] starting ...")
    url = resource["url"]
    index = resource["index"]
    index_exists: bool = check_index_existance(index)

    local_path = f"./{index}-{str(uuid.uuid1())}.txt"

    response = requests.get(url)

    with open(local_path, "wb") as f:
        f.write(response.content)

    loader = TextLoader(local_path)
    doc = loader.load()
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100
    )
    chunks = text_splitter.split_documents(doc)
    embeddings = OpenAIEmbeddings()

    if not index_exists:
        index_db = FAISS.from_documents(chunks, embedding=embeddings)
    else:
        index_db = FAISS.load_local(index, embeddings=embeddings)
        index_db.add_documents(chunks)

    index_db.save_local(index)
    logger.info(f"Index {index} saved locally! {index_db.docstore._dict}")


def download_html_file_and_ingest(resource):
    logger.info(f"[[download_html_file_and_ingest]] starting ...")
    url = resource["url"]
    index = resource["index"]
    index_exists: bool = check_index_existance(index)

    local_path = f"./{index}-{str(uuid.uuid1())}.html"

    response = requests.get(url)
    with open(local_path, "wb") as f:
        f.write(response.content)
    logger.info(f"File downloaded successfully to {local_path} !")

    loader = WebBaseLoader(url)
    doc = loader.load()
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100
    )
    chunks = text_splitter.split_documents(doc)
    embeddings = OpenAIEmbeddings()
    if not index_exists:
        index_db = FAISS.from_documents(chunks, embedding=embeddings)
    else:
        index_db = FAISS.load_local(index, embeddings=embeddings)
        index_db.add_documents(chunks)

    index_db.save_local(index)
    logger.info(f"Index {index} saved locally! {index_db.docstore._dict}")


def custom_format(text):
    """Given a string with multiple sentences, add newlines between each sentence."""
    no_greet = text.replace("¡Hola! ", "")
    sentences = no_greet.split(". ")  # split the text into sentences
    spaced = "\n\n".join(
        sentences
    )  # join the sentences with newline characters
    return {"spaced": spaced, "sentences": sentences}


async def ask(
    user_input,
    index,
    company_name,
    personality,
    language,
    default_response,
    emoji_level,
):
    db = FAISS.load_local(index, embeddings=OpenAIEmbeddings())
    source_retrieval_start = time.time()
    sources = db.similarity_search_with_score(user_input)
    first_source = sources[0][0].page_content
    source_metadata = sources[0][0].metadata
    retrieval_score = str(round(sources[0][1], 2))
    source_retrieval_end = time.time()
    emoji_prompt = emoji_prompter(emoji_level)
    source_retrieval_time = source_retrieval_end - source_retrieval_start
    prompt_12 = f"""
                You are the best assistant of an organization named '{company_name}'. Answer the user's question using ONLY the data between the three backticks. Don't try to make up an answer.
        Go through these steps:
        - Step 1: Read the DATA and the user's question, if there is no relation between the data and the question just respond with ```{default_response}```. 
        - Step 2: Create a very short answer according to the TONE and in a way that is very easy to read in a chat conversation. {emoji_prompt}.

        DATA: ```{first_source}```

        TONE: ```{personality}```

        """

    system_prompt = prompt_12
    user = user_input
    completion = await system_user_v1_turbo_t0_full(
        system=system_prompt, user=user
    )
    # return add_newlines(response)
    return {
        "system_prompt": system_prompt,
        "source_retrieval_time": source_retrieval_time,
        "retrieval_score": retrieval_score,
        "source_metadata": source_metadata,
        **completion,
    }


local_file = "/Users/alexander/clients/yzn/gpt-api/app/docs/osde.txt"


# logger.critical(f"Number of chunks: {len(chunks)}")
# logger.critical(f"first chunk: {chunks[0]}")
# logger.critical(f"second chunk: {chunks[1]}")


async def faiss_ingest(resources):
    logger.info("[[ faiss_kb ]] faiss_ingest starting ...")
    try:
        for resource in resources:
            logger.info(f"[[ faiss_kb ]] Ingesting {resource} ")
            ingest_resource(resource)
    except Exception as e:
        logger.error(f"[[ Error ingesting resources: {e} ]]")
        return {"status": "ERROR", "message": f"{e}"}
    return {"status": "READY_TO_ANSWER"}


async def faiss_retrieval(
    user_input: str,
    index: str = "main",
    ranking_size: int = 3,
):
    db = FAISS.load_local(index, embeddings=OpenAIEmbeddings())
    source_retrieval_start = time.time()
    sources = db.similarity_search_with_score(user_input)
    logger.info(f"[[retrieve]] sources: {min(ranking_size,len(sources))}")
    logger.info(
        "[[retrieve]] sources:"
        f" {sources[min(ranking_size,len(sources))][0].page_content}"
    )
    ranking_limit = min(ranking_size, len(sources))
    chunks = [sources[i][0].page_content for i in range(0, ranking_limit)]
    metadatas = [sources[i][0].metadata for i in range(0, ranking_limit)]
    scores = [
        str(round(1 - sources[i][1], 2)) for i in range(0, ranking_limit)
    ]
    source_retrieval_end = time.time()
    source_retrieval_time = source_retrieval_end - source_retrieval_start
    return {
        "chunks": chunks,
        "metadatas": metadatas,
        "scores": scores,
        "source_retrieval_time": round(source_retrieval_time, 2),
    }


async def faiss_ask_retrieval(
    user_input: str,
    session_id: str,
    personality: str,
    company_name: str,
    language: str,
    default_response: str,
    index: str,
    emoji_level: int,
):
    start_time = time.time()
    answer = await ask(
        user_input=user_input,
        index=index,
        company_name=company_name,
        language=language,
        personality=personality,
        default_response=default_response,
        emoji_level=emoji_level,
    )
    # transform answer["openai_processing_ms"] into an integer
    openai_process_ms_int = int(answer["openai_processing_ms"])
    openai_processing_time = openai_process_ms_int / 1000

    end_time = time.time()
    execution_time = end_time - start_time
    formatted = custom_format(answer["content"])
    source_retrieval_time = answer["source_retrieval_time"]
    processing_time = (
        execution_time - source_retrieval_time - openai_processing_time
    )
    response = {
        "answer": formatted["spaced"],
        "sentences": formatted["sentences"],
        "usage": answer["usage"],
        "execution_time": round(execution_time, 2),
        "openai_processing_time": round(openai_processing_time, 2),
        "system_prompt": answer["system_prompt"],
        "source_retrieval_time": round(source_retrieval_time, 2),
        "processing_time": round(processing_time, 2),
        "retrieval_score": answer["retrieval_score"],
        "source_metadata": answer["source_metadata"],
    }
    logger.info(
        "[[FAISS ASK RETRIEVAL]] :: TIMES ["
        f" Total:{response['execution_time']} |"
        f" Retrieval:{response['source_retrieval_time']} |"
        f" OpenAI:{response['openai_processing_time']} |"
        f" Api:{response['processing_time']} ]"
        f" :: INPUT: {user_input}"
        " :: TOKENS: ["
        f" Prompt: {response['usage']} ]"
        # f" Completion: {response['usage']['completion_tokens']} |"
        # f" Total: {response['usage']['total_tokens']} ]"
    )
    return response


def emoji_prompter(level):
    if level == 0:
        return "Never use emojis"
    if level == 1:
        return "Use emojis sparingly"
    if level == 2:
        return "Use emojis moderately"
    if level == 3:
        return "Use emojis liberally"
    if level == 4:
        return "Use emojis excessively and in every sentence"
