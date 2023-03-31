from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
from tools.openaidirect import system_completion_v1_turbo_t0
import json


load_dotenv()
completion_url = os.getenv("COMPLETION_URL")


async def classify_2_emotions_score_lc(user_input: str):
    system_prompt = "extract 2 emotions. emotion name in spanish. add %"
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"text:{user_input}"),
    ]
    chat = ChatOpenAI(verbose=True, temperature=0)
    openai_response = chat(messages)
    # response = json.loads(openai_response.content)
    return openai_response.content


async def classify_2_emotions_score_d(emotions: list, user_input: str):
    system_prompt_1 = (
        f"emotions: {emotions}. text: `{user_input}`. pick 2 emotions from the"
        " list."
    )
    selected_emotions = system_completion_v1_turbo_t0(system_prompt_1)
    system_prompt_2 = (
        f"emotions: {selected_emotions}. text: `{user_input}`. return only"
        " score of each emotion in a valid json"
    )
    scored_emotions = await system_completion_v1_turbo_t0(system_prompt_2)
    emotions = json.loads(scored_emotions).items()
    emotion_list = [{"name": k, "score": v} for k, v in emotions]

    response = {
        "emotions": emotion_list,
    }
    return response
