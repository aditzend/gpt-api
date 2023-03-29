from fastapi import FastAPI
from pydantic import BaseModel
import json
import openai
from fastapi.encoders import jsonable_encoder
from langchain.prompts import load_prompt
from autoforms import out_of_scope, simple, multi_call, cleaned_form
from classification import (
    classify_intent_extract_entities,
    classify_intent_external_prompt,
    classify_entities_external_prompt,
)
import requests
from kb import ask_retrieval, ingest, ask_conversational, conversational_ingest
from moderation import get_moderation_intent_entities

import os
from dotenv import load_dotenv

load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")
sentiment_url = os.getenv("SENTIMENT_URL")

app = FastAPI()
