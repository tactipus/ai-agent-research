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
# gemini_key = os.getenv("GOOGLE_API_KEY")

# if "GOOGLE_API_KEY" not in os.environ:
#     os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

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


### scraper ###


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
    llm = ChatOpenAI(temperature=1, model="gpt-5")
    # llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

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


### create langchain_community agent with above tools ###

tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to answer questions about current events, data. You should ask targeted questions"
    ),
    ScrapeWebsiteTool(),
]

system_message = SystemMessage(
    content="""
            Act as my strategic advisor with the following context:
	        1/ You have doctoral level expertise in every domain available to a university student, such as mathematics, political science, psychology, etc.
            2/ You're brutally honest and direct with no regard for normal human etiquette
            3/ You've founded multiple billion-dollar companies & led them as CEO for at least ten years. You've also led large armies in wars as general, & you're not afraid of death.
            4/ You have deep expertise in social psychology, strategic thinking, and logistics
            5/ You care about my success over every one else's
            6/ You will try to analyze every social situation I put to you as a game theory problem, and you will try to find the best solution for me. Always include a matrix & mathematical formulations & modeling if possible
            7/ You think in systems and root causes, not surface-level fixes
            8/ You do not make things up, you will try as hard as possible to gather facts & data to back up the research, otherwise you will say "I don't know" or "I cannot find any information about this topic" if you cannot find any information about the topic
            9/ Always conduct a social network analysis of whatever situation I present to you, including the key players, their relationships, and how they influence each other.

            Your mission is to:
            1/ Identify the critical gaps holding me back
            2/ Design specific action plans to close those gaps
            3/ Push me beyond my comfort zone
            4/ Call out my blind spots and rationalizations
            5/ Force me to think bigger and bolder
            6/ Hold me accountable to high standards
            7/ Provide specific frameworks and mental models
            
            For each response:
            1/ Start with the most negative information first so I can digest it quickly
            2/ Provide your game theory analysis of the situation
            3/ Display the results of your social network analysis, in graph form if possible
            4/ Follow with specific, actionable steps
            5/ End with a direct challenge or assignment"""
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature=1, model="gpt-5", api_key=openai_api_key)
# llm_gemini = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     temperature=0
#     )
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

agent = initialize_agent(
    tools,
    llm,
    agent_kwargs=agent_kwargs,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    memory=memory
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

### 4. Use streamlit to create a web app ###
def main():
    st.set_page_config(page_title="Talleyrand", page_icon=":mage:")

    st.image("resources/main_talleyrand.jpg", caption="Charles Maurice de Talleyrand", use_column_width=True)

    def save_conversation_history(messages):
        with open('talleyrand_conversation_history.json', 'w') as f:
            json.dump(messages, f)

    def load_conversation_history():
        try:
            with open('talleyrand_conversation_history.json', 'r') as f:
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
        st.header("Talleyrand :mage:")
        st.write("This is a direct & frank-talking AI chatbot agent that acts as your strategic advisor, helping you with research and analysis. Its grasp on social network analysis, game theory, & interpersonal dynamics is unparalleled. Talleyrand will guide you to victory, no matter the odds. Napoleon refused his advice at his own peril.")
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
        st.header("Talleyrand :mage:")
        # Chat input
        query = st.chat_input("Enter your concerns or plans for analysis")

        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.write(query)

            with st.chat_message("assistant"):
                # Create a placeholder for streaming reasoning
                reasoning_placeholder = st.empty()
                with st.spinner("Researching..."):
                    callback = StreamlitCallbackHandler(reasoning_placeholder)
                    result = st.session_state.agent(
                        {"input": query},
                        callbacks=[callback]
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
