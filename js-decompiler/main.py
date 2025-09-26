import os
from dotenv import load_dotenv
import getpass

from langchain_core.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType
# from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, SecretStr
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage
from fastapi import FastAPI
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks.base import BaseCallbackHandler

openai_api_key: SecretStr = os.getenv("OPENAI_API_KEY") # type: ignore

def get_js_file():
    file_name = os.path
    return file_name


class DecompilerInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    path: str = Field(description="The path of the javascript file")

class FileGetterTool(BaseTool):
    name: str = "scrape_website"
    description: str = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = DecompilerInput

    def _arun(self, url: str):
        raise NotImplementedError("error here")

tools = [
    Tool(
        name="Get JS File",
        func=get_js_file,
        description="gets js file that needs to be decompiled"
    ),
    FileGetterTool,
]

agent_kwargs = {
    "press": [MessagesPlaceholder(variable_name="memory")],
}

llm = ChatOpenAI(
    temperature=1, 
    model="gpt-4o-mini", 
    api_key=openai_api_key)

agent = initialize_agent(
    tools,
    llm,
    agent_kwargs=agent_kwargs,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    handle_parsing_errors=True
)

def user_pushed_button():
    return True

def main():
    input = user_pushed_button()

    if input:
        st.session_state.messages.append({"role": "user", "content": input})
        with st.chat_message("user"):
            st.write(input)

        with st.chat_message("assistant"):
            # Create a placeholder for streaming reasoning
            reasoning_placeholder = st.empty()
            with st.spinner("Decompiling..."):
                result = st.session_state.agent(
                    {"input": input},
                )
                st.write(result['output'])
                st.session_state.messages.append({"role": "assistant", "content": result['output']})




if __name__ == "__main__":
    main()

app = FastAPI()

@app.post("/")
def bountyAgent(query: Query):
    query = query.query # type: ignore
    content = agent({"input": query})
    actual_content = content['output']
    return actual_content