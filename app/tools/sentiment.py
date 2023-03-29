from fastapi import FastAPI
from pydantic import BaseModel
import json
import openai
from fastapi.encoders import jsonable_encoder
from langchain.prompts import load_prompt

import requests
from dotenv import load_dotenv
import os

load_dotenv()

sentiment_url = os.getenv("SENTIMENT_URL")


async def sentiment_emotion_hate(text):
    endpoint = sentiment_url + "/full/"
    response = requests.post(endpoint, json={"text": text})
    return response.json()
