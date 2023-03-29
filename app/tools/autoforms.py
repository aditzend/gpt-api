from langchain.prompts import load_prompt
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage


import json
import os

form_with_out_of_scope_template = PromptTemplate(
    input_variables=[
        "user_input",
        "slots_to_fill",
        "slots_filled",
        "personality",
    ],
    template="""
user_input =  {user_input}
slots_to_fill =  {slots_to_fill}
slots_filled = {slots_filled}

Act as a company chatbot in charge of obtaining specific data from a user.

Data will be stored in slots.

First, analyze the message in user_input and decide if it is out of scope.
If it is out of scope, ask the user to stay on topic and return to providing
information about the slots_to_fill.

If you think the user needs to take a step back
  and is asking for
useful information that sits in the company's knowledge base, set the
status variable to "OOS" and return the answer.

Analyze the user_input and check if there is a slot_to_fill value in it.

If you find one or many of them, store them all in the slots_filled list.

Compare the slots_filled list with the slots_to_fill list.

If all the required slots in slots_to_fill are present in slots_filled,
follow these steps:
1. Set the status variable to "IN_PROGRESS"
2. 
predict the 'utterance' key with an enumeration of the slots and set status
variable to "FINISHED".

If there are still slots_to_fill that are not in slots_filled,
follow these steps with slots_still_missing:
1. Set the status variable to "IN_PROGRESS"
2. Choose one slot in slots_still_missing 
3. Create an appropiate question using the slot description
    and these personality guidelines= {personality}.
      
Always calculate the confidence of your prediction and store it in the 
'confidence' key.

Think step by step and be very careful with your predictions.

Just output a valid JSON object, no pretext and no posttext,
with the following structure: 

"status",
"confidence",
"question",
"slots_filled",
"slots_to_fill",


Go!
    """,
)

explained_template = PromptTemplate(
    input_variables=[
        "user_input",
        "slots_to_fill",
        "slots_filled",
    ],
    template="""
    user_input:  {user_input}
slots_to_fill:  {slots_to_fill}

slots_filled: {slots_filled}

status: "IN_PROGRESS"

Use the following questions to guide your reasoning, do not answer them.
Does the user give any value needed in slots_to_fill list?
Is there any other slot in slots_to_fill to ask for or are we finished? If we are finished change the status to "DONE".

Is the user input unrelated to the slot filling process? If so change the status to "OUT_OF_SCOPE".

Update the slots_filled list, if a value is none, don't return it in the slots_filled list.

If there is one or more slots left to fill, think of only one good question to ask for all of them and store it in the question variable.

If there are no more slots to fill thank the user for giving all the requested information and enumerate all values of the filled slots in the question variable.

Calculate the confidence key indicating form 0.0 to 1.0 the level of confidence of your decisions.

Return a response variable which is a valid JSON object with all the variables. 
""",
)

extractor_template = PromptTemplate(
    input_variables=["explained", "personality"],
    template="""
    Extract the response as a valid JSON object: 

    {personality}

    ###

    {explained}
    """,
)

simple_form = PromptTemplate(
    input_variables=[
        "user_input",
        "slots_to_fill",
        "slots_filled",
        "personality",
    ],
    template="""
user_input =  {user_input}
slots_to_fill =  {slots_to_fill}
slots_filled = {slots_filled}
personality = {personality}
    """,
)


async def out_of_scope(user_input, slots_to_fill, slots_filled, personality):
    llm = OpenAI(temperature=0.0)

    prompt = form_with_out_of_scope_template.format(
        user_input=user_input,
        slots_to_fill=slots_to_fill,
        slots_filled=slots_filled,
        personality=personality,
    )
    response = llm(prompt)
    clean = "{" + response.split("{", 1)[1]
    clean = clean.replace("'", '"')
    print(clean)
    response = json.loads(clean)
    return response


async def simple(user_input, slots_to_fill, slots_filled, personality):
    llm = OpenAI(temperature=0.0)

    prompt = simple_form.format(
        user_input=user_input,
        slots_to_fill=slots_to_fill,
        slots_filled=slots_filled,
        personality=personality,
    )
    response = llm(prompt)
    clean = "{" + response.split("{", 1)[1]
    clean = clean.replace("'", '"')
    print(clean)
    response = json.loads(clean)
    return response


async def multi_call(user_input, slots_to_fill, slots_filled, personality):
    chat = ChatOpenAI(verbose=True, temperature=0)

    prompt_1 = explained_template.format(
        user_input=user_input,
        slots_to_fill=slots_to_fill,
        slots_filled=slots_filled,
    )
    messages_1 = [
        HumanMessage(content=prompt_1),
    ]

    response_1 = chat(messages_1)
    explained = response_1.content

    print(explained)
    prompt_2 = extractor_template.format(
        explained=explained,
        personality=personality,
    )
    messages_2 = [
        HumanMessage(content=prompt_2),
    ]
    response_2 = chat(messages_2)
    response = response_2.content
    print(response)

    # response = json.loads(response)
    return response


async def cleaned_form(user_input, slots_to_fill, slots_filled, personality):
    chat = ChatOpenAI(verbose=True, temperature=0)
    prompt_1 = explained_template.format(
        user_input=user_input,
        slots_to_fill=slots_to_fill,
        slots_filled=slots_filled,
    )
    messages_1 = [
        HumanMessage(content=prompt_1),
    ]

    response_1 = chat(messages_1)
    explained = response_1.content
    clean = "{" + explained.split("{", 1)[1]
    clean = clean.replace("'", '"')
    print(clean)
    response = json.loads(clean)
    return response
