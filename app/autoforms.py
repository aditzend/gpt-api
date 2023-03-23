from langchain.prompts import load_prompt
from langchain import PromptTemplate
from langchain.llms import OpenAI
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


async def out_of_scope(user_input, slots_to_fill, slots_filled, personality):
    llm = OpenAI(temperature=0.9)

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
