import requests
import os
from dotenv import load_dotenv

load_dotenv()
completion_url = os.getenv("COMPLETION_URL")
openai_api_key = os.getenv("OPENAI_API_KEY")
authorization = f"Bearer {openai_api_key}"


async def system_completion_v1_turbo_t0(content: str):
    headers = {
        "Content-Type": "application/json",
        "Authorization": authorization,
    }
    payload = {
        "model": "gpt-3.5-turbo",
        "temperature": 0,
        "messages": [
            {
                "content": content,
                "role": "system",
            }
        ],
    }
    print(content)
    raw = requests.post(completion_url, headers=headers, json=payload)
    response = raw.json()
    print(response["choices"][0]["message"]["content"])
    response = response["choices"][0]["message"]["content"]
    return response


async def system_user_v1_turbo_t0(system: str, user: str):
    headers = {
        "Content-Type": "application/json",
        "Authorization": authorization,
    }
    payload = {
        "model": "gpt-3.5-turbo",
        "temperature": 0,
        "messages": [
            {
                "content": system,
                "role": "system",
            },
            {
                "content": user,
                "role": "user",
            },
        ],
    }
    raw = requests.post(completion_url, headers=headers, json=payload)
    response = raw.json()
    print(response)
    response = response["choices"][0]["message"]["content"]
    return response


async def system_user_v1_turbo_t0_full(system: str, user: str):
    headers = {
        "Content-Type": "application/json",
        "Authorization": authorization,
    }
    payload = {
        "model": "gpt-3.5-turbo",
        "temperature": 0,
        "max_tokens": 200,
        "messages": [
            {
                "content": system,
                "role": "system",
            },
            {
                "content": user,
                "role": "user",
            },
        ],
    }
    raw = requests.post(completion_url, headers=headers, json=payload)
    response = raw.json()
    content = None
    try:
        content = response["choices"][0]["message"]["content"]
    except KeyError:
        print("The key 'choices' is not present in the response")

    try:
        usage = response["usage"]
    except KeyError:
        print("Usage not present")
    return {
        "content": content,
        "usage": usage,
        "openai_processing_ms": raw.headers["openai-processing-ms"],
    }
