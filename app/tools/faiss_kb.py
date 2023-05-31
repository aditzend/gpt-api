import time
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from tools.openaidirect import system_user_v1_turbo_t0_full
from ingestors.csv_ingestors import download_csv_file_and_ingest
from ingestors.docx_ingestors import download_docx_file_and_ingest
from ingestors.html_ingestors import download_html_file_and_ingest
from ingestors.pdf_ingestors import download_pdf_file_and_ingest
from ingestors.youtube_ingestors import youtube_ingest
from ingestors.image_ingestors import (
    download_jpg_file_and_ingest,
    download_png_file_and_ingest,
)
from ingestors.site_ingestors import ingest_site
from ingestors.pptx_ingestors import download_pptx_file_and_ingest
from ingestors.txt_ingestors import download_txt_file_and_ingest
import logging
from dotenv import load_dotenv
import os
import requests
import json

# import sseclient

logger = logging.getLogger("uvicorn")

load_dotenv()


def ingest_resource(resource):
    if resource["type"] == "html":
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
    if resource["type"] == "site":
        ingest_site(resource)


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
    db = FAISS.load_local(f"indexes/{index}", embeddings=OpenAIEmbeddings())
    source_retrieval_start = time.time()
    sources = db.similarity_search_with_score(user_input)
    first_source = sources[0][0].page_content
    second_source = ""
    if len(sources) > 1:
        second_source = sources[1][0].page_content
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

        DATA: ```{first_source + second_source}```

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


async def faiss_ingest(resources):
    logger.debug("[[ faiss_kb faiss_ingest ]] Starting ...")
    try:
        # Deleting previuos indexes
        for resource in resources:
            # delete index of resource
            # delete main index
            logger.debug(
                "[[ faiss_kb faiss_ingest ]] Deleting index"
                f" {resource['index']}"
            )
        for resource in resources:
            logger.debug(f"[[ faiss_kb faiss_ingest ]] Ingesting {resource} ")
            ingest_resource(resource)
    except Exception as e:
        logger.error(
            f"[[ faiss_kb faiss_ingest  ]] Error ingesting resources: {e}"
        )
        return {"status": "ERROR", "message": f"{e}"}
    return {"status": "READY_TO_ANSWER"}


async def faiss_retrieval(
    user_input: str,
    index: str = "main",
    ranking_size: int = 3,
):
    INDEXES_PATH = os.getenv("INDEXES_PATH")
    db = FAISS.load_local(
        f"{INDEXES_PATH}/{index}", embeddings=OpenAIEmbeddings()
    )
    source_retrieval_start = time.time()
    sources = db.similarity_search_with_score(user_input)
    ranking_limit = min(ranking_size, len(sources))
    logger.debug(
        f"[[retrieve]] sources: {len(sources)} ranking_size:"
        f" {ranking_size} ranking_limit: {ranking_limit}"
    )

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


API_KEY = os.getenv("OPENAI_API_KEY") or "OPENAI_API_KEY"


# async def faiss_ask_streaming():
#     reqUrl = "https://api.openai.com/v1/chat/completions"
#     reqHeaders = {
#         "Accept": "text/event-stream",
#         "Authorization": "Bearer " + API_KEY,
#     }
#     reqBody = {
#         "model": "gpt-3.5-turbo",
#         "messages": [{"role": "user", "content": "cual es el mejor cafe?"}],
#         "max_tokens": 200,
#         "temperature": 1,
#         "stream": True,
#     }
#     request = requests.post(
#         reqUrl, stream=True, headers=reqHeaders, json=reqBody
#     )
#     client = sseclient.SSEClient(request)
#     logger.debug(client)
#     response = ""
#     for event in client.events():
#         if (event.data != "[DONE]") and (
#             "content" in json.loads(event.data)["choices"][0]["delta"]
#         ):
#             response += json.loads(event.data)["choices"][0]["delta"][
#                 "content"
#             ]
#             print(
#                 json.loads(event.data)["choices"][0]["delta"]["content"],
#                 end="",
#                 flush=True,
#             )
#             # if "." in response:
#     return response
