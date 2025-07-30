## Conversational Q&A Chatbot
import streamlit as st
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

## Streamlit UI
st.set_page_config(page_title="Conversational Q&A Chatbot")
st.header("Hey, Let's Chat")

load_dotenv()

chat = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="LLaMA3-70b-8192"
)

# Custom system prompt (You can modify this prompt to suit tone/style/accuracy)
system_prompt = """
You are a friendly, intelligent, and confident AI assistant. 
Engage in clear and thoughtful conversations while providing accurate, helpful answers.
Keep your tone warm, professional, and conversational — like a helpful expert who enjoys making things easy to understand.
Avoid vague language and unnecessary disclaimers. If something is unknown or unclear, politely say “I’m not sure about that” instead of guessing.
Keep responses concise but informative, and aim to make the conversation feel natural and helpful.

"""

# Initialize session state with improved prompt
if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages'] = [
        SystemMessage(content=system_prompt.strip())
    ]

## Function to get response from model
def get_chatmodel_response(question):
    st.session_state['flowmessages'].append(HumanMessage(content=question))
    answer = chat(st.session_state['flowmessages'])
    st.session_state['flowmessages'].append(AIMessage(content=answer.content))
    return answer.content

# User input
input = st.text_input("Input:", key="input")
submit = st.button("Ask the question")

## When button is clicked
if submit and input:
    response = get_chatmodel_response(input)
    st.subheader("The Response is")
    st.write(response)
