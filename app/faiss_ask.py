from dotenv import load_dotenv
import os
from langchain.embeddings.openai import OpenAIEmbeddings

import time

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from tools.openaidirect import (
    system_completion_v1_turbo_t0,
    system_user_v1_turbo_t0,
)

from langchain.document_loaders import TextLoader

load_dotenv()


def add_newlines(text):
    """Given a string with multiple sentences, add newlines between each sentence."""
    sentences = text.split(".")  # split the text into sentences
    new_text = "\n".join(
        sentences
    )  # join the sentences with newline characters
    return new_text


async def ask(query, db):
    sources = db.similarity_search(query)
    first_source = sources[0].page_content
    system = (
        f"DATA: {first_source}\n\n###Si no sabes la respuesta, responde con un"
        " '0'.Tu respuesta tiene que ser fácil de leer, usa un emoji"
        " diferente para cada oración de acuerdo a su contenido y deja dos"
        " líneas vacías entre cada oración."
        # If you don't know the answer, just respond with"
        # " '0'. Make the answer easy to read, use bulletpoints, different"
        # " emojis and line breaks. Leave a blank line between each paragraph."
    )
    user = query
    response = await system_user_v1_turbo_t0(system=system, user=user)
    # return add_newlines(response)
    return response


local_file = "/Users/alexander/clients/yzn/gpt-api/app/docs/osde.txt"

loader = TextLoader(local_file)
osdedoc = loader.load()
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=0
)
chunks = text_splitter.split_documents(osdedoc)


embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(chunks, embeddings)


async def faiss_ask_retrieval(user_input, session_id, personality):
    start_time = time.time()

    answer = await ask(user_input, db)

    end_time = time.time()
    execution_time = end_time - start_time
    return {"answer": answer, "execution_time": execution_time}
