from backend import *
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("openai_api_key")
os.environ['llm']=os.getenv("OpenAI_model")

llm = ChatOpenAI(temperature=0, model=os.getenv("OpenAI_model"))

directory_path = "folder"

def chatbot_interface(query):
    try:
        if query == '':
            return ''
        agent = admin_agent(directory_path,llm)
        response = agent({"input":f"{query}"})
        return response['output']
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred while processing your request."
    
def handler(query):
    try:
        if query == '':
            return ''
        agent = admin_agent(directory_path,llm)
        response = agent({"input":f"{query}"})
        return response['output']

    except Exception as e:
        return st.error(f"An error occurred: {e}")

st.title("Chat with docs")

user_query = st.text_input("You:")

if user_query:
  response = handler(user_query)
  st.write("Assistant:", response)