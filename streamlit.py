import streamlit as st
import openai_api
import json
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = openai_api.key()

# Load data
df = pd.read_json('ICLR_2019_mini_optimized.json')

# Create agent
agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model='gpt-3.5-turbo-16k-0613'),  # Define model
    df,  # Data frame
    verbose=True,  # Inference process output
    agent_type=AgentType.OPENAI_FUNCTIONS,  # Agent type
)

# Streamlit UI
st.set_page_config(page_title="Chatbot Demo", page_icon=":robot:")
st.title("Chatbot Demo")

# User input
user_input = st.text_input("You:", "")

# Chat logic
if user_input:
    # Run agent with user input
    response = agent.run(user_input)

    # Display assistant's response
    st.text(f"Assistant: {response}")

# Example queries
st.markdown("### Example Queries:")
st.code("How many rows are there in the dataset?")
st.code("Summarize the paper titled \"Layerwise Recurrent Autoencoder for General Real-world Traffic Flow Forecasting\"")
st.code("Which topics were popular in the dataset?")
st.code("What is the survival rate of male passengers? Please tell me in %")
