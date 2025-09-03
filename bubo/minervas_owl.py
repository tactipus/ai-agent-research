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
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage
from fastapi import FastAPI
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks.base import BaseCallbackHandler

load_dotenv()
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
ghidra_server_url = os.getenv("GHIDRA_SERVER_URL")


# search tool

def search(query):
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query
    })
    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)

    return response.text


# scraper

def scrape_website(objective: str, url: str):
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the request
    data = {
        "url": url
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)

    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENT:", text)

        if len(text) > 10000:
            output = summary(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")


def summary(objective, content):
    llm = ChatOpenAI(temperature=1, model="gpt-4o-mini")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output

class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")

class ScrapeWebsiteTool(BaseTool):
    name: str = "scrape_website"
    description: str = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")

# create langchain_community agent with above tools

tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to answer questions about current events, data. You should ask targeted questions"
    ),
    ScrapeWebsiteTool(),
]

system_message = SystemMessage(
    content="""You are a world class researcher at Cornell University who can do detailed research on any topic and produce fact-based results, hence why you got a doctorate in Computer Science; 
            you do not make things up, you will try as hard as possible to gather facts & data to back up the research, otherwise you will say "I don't know" or "I cannot find any information about this topic" if you cannot find any information about the topic. You will also have a dorky & witty style along with being a female scientist, & you will also have lots of confidence. You will be direct & you will not be sycophantic.
            
            You also have access to Ghidra reverse engineering tools and can analyze binary files, decompile functions, examine assembly code, and trace program execution flow. Use these tools when users ask about binary analysis, malware analysis, or reverse engineering tasks.
            
            Please make sure you complete the objective above with the following rules:
            1/ You should do enough research to gather as much information as possible about the objective
            2/ If there are URLs of relevant links & articles, you will scrape them to gather more information
            3/ After scraping & searching, you should think "is there any new things I should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this for more than 3 iterations
            4/ You should not make things up, you should only write facts & data that you have gathered
            5/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
            6/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
            7/ While you're doing the research tasks, you should keep me aware of your thought process each step of the way, so I can understand what you're doing and why you're doing it by displayin it on the chat screen in streamlit
            8/ When using Ghidra tools, explain what you're analyzing and why it's relevant to the research objective"""
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(
    temperature=1, 
    model="gpt-4o-mini", 
    api_key=openai_api_key)

memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

agent = initialize_agent(
    tools,
    llm,
    agent_kwargs=agent_kwargs,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)

class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.thoughts = []

    def on_chain_start(self, *args, **kwargs):
        self.thoughts = []

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.thoughts.append(f"🔎 Tool: {serialized['name']} | Input: {input_str}")
        self.placeholder.markdown("\n\n".join(self.thoughts))

    def on_tool_end(self, output, **kwargs):
        self.thoughts.append(f"✅ Output: {output}")
        self.placeholder.markdown("\n\n".join(self.thoughts))

    def on_text(self, text, **kwargs):
        self.thoughts.append(f"💡 {text}")
        self.placeholder.markdown("\n\n".join(self.thoughts))

# 4. Use streamlit to create a web app
def main():
    st.set_page_config(page_title="minerva's owl", page_icon=":bird:")

    def save_conversation_history(messages):
        with open('conversation_history.json', 'w') as f:
            json.dump(messages, f)

    def load_conversation_history():
        try:
            with open('conversation_history.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    # Initialize session state for chat history if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = load_conversation_history()
    if "agent" not in st.session_state:
        st.session_state.agent = agent

    MainTab, HistoryTab, InfoTab = st.tabs(["Main", "History", "Info"])

    with InfoTab:
        st.header("minerva's owl :bird:")
        st.write("something about minerva & owls & wisdom...")
        st.write("It uses LangChain, OpenAI, and Google AI to gather information and provide insights. It also has a persistent memory that allows it to remember your previous conversations.")

    with HistoryTab:
        st.header("Conversation History")
        if st.session_state.messages:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
        else:
            st.write("No conversation history found.")
        # Add a clear button
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.session_state.agent = agent
            save_conversation_history([])  # Save empty history when clearing
            st.rerun()

    with MainTab:
        st.header("Bubo :owl:")
        # Chat input
        query = st.chat_input("Would you kindly share your research goals with me?")

        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.write(query)

            with st.chat_message("assistant"):
                # Create a placeholder for streaming reasoning
                # reasoning_placeholder = st.empty()
                with st.spinner("Researching..."):
                    # callback = StreamlitCallbackHandler(reasoning_placeholder)
                    result = st.session_state.agent(
                        {"input": query},
                        # callbacks=[callback]
                    )
                    st.write(result['output'])
                    st.session_state.messages.append({"role": "assistant", "content": result['output']})
                    save_conversation_history(st.session_state.messages)


if __name__ == '__main__':
    main()


# 5. Set this as an API endpoint via FastAPI
app = FastAPI()


class Query(BaseModel):
    query: str


@app.post("/")
def researchAgent(query: Query):
    query = query.query
    content = agent({"input": query})
    actual_content = content['output']
    return actual_content

# Add these new functions for persistent memory
