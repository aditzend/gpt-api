import asyncio
from aiohttp import ClientSession
import os
from dotenv import load_dotenv
import time

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
authorization = f"Bearer {openai_api_key}"
load_dotenv()
completion_url = os.getenv("COMPLETION_URL")


async def fetch(url, session):
    headers = {
        "Content-Type": "application/json",
    }
    async with session.get(url=url, headers=headers) as response:
        delay = response.headers.get("DELAY")
        date = response.headers.get("DATE")
        print("{}:{} with delay {}".format(date, response.url, delay))
        return await response.text()


async def moderation(url, session):
    model_start_time = time.time()
    payload = {"input": "Te como a besos en la c"}

    async with session.post(url, json=payload) as response:
        answer = await response.json()
        openai_time = response.headers["openai-processing-ms"]
        model_end_time = time.time()
        model_elapsed_time = model_end_time - model_start_time
        print(
            f"Model elapsed time: {model_elapsed_time}, openai time:"
            f" {openai_time} ms "
            f" first model: {answer['results'][0]['flagged']}"
        )


async def intent(url, session):
    models_url = "https://api.openai.com/v1/models"
    model_start_time = time.time()
    payload = {"input": "Te como a besos en la c"}
    async with session.get(url=models_url, json=payload) as response:
        answer = await response.json()
        openai_time = response.headers["openai-processing-ms"]
        model_end_time = time.time()
        model_elapsed_time = model_end_time - model_start_time
        print(
            f"Model elapsed time: {model_elapsed_time}, openai time:"
            f" {openai_time} ms "
            f" first model: {answer['data'][0]['id']}"
        )


async def moderation_fetch(sem, url, session):
    # Getter function with semaphore.
    async with sem:
        await moderation(url, session)


async def intent_fetch(sem, url, session):
    # Getter function with semaphore.
    async with sem:
        await intent(url, session)


async def run(r):
    url = "http://66.97.45.66:39500/v1/.well-known/live"
    tasks = []
    # create instance of Semaphore
    sem = asyncio.Semaphore(3)
    start_time = time.time()
    # Create client session that will ensure we dont open new connection
    # per each request.
    headers = {
        "Content-Type": "application/json",
        "Authorization": authorization,
    }
    async with ClientSession(headers=headers) as session:
        for i in range(r):
            # pass Semaphore and session to every GET request
            moderation = asyncio.ensure_future(
                moderation_fetch(
                    sem, "https://api.openai.com/v1/moderations", session
                )
            )
            tasks.append(moderation)

            # intent = asyncio.ensure_future(
            #     intent_fetch(sem, url.format(i), session)
            # )
            # tasks.append(intent)

        responses = asyncio.gather(*tasks)
        await responses
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time}")


number = 3
asyncio.run(run(number))
