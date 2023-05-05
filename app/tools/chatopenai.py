import os
import time


from dotenv import load_dotenv

from langchain.chat_models import ChatAnthropic
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

load_dotenv()


chat = ChatAnthropic()
start_time = time.time()

# your process goes here
messages = [
    HumanMessage(
        content=(
            " [Asco, Desaprobación,Tristeza]."
            " text: `no puedo mas, no quiero ni despertarme`."
            " return only score of each emotion in a valid json"
        )
    )
]
res = chat(messages)

print(res.content)

end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")
