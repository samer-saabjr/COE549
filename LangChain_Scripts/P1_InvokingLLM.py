# Import required modules from langchain
from langchain_openai import ChatOpenAI  # For interacting with Google's OpenAI model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage  # Message types for chat

import os
from dotenv import load_dotenv

load_dotenv()

# Function to create a OpenAI LLM using type hints
def create_openai_llm(model_name: str = "gpt-5.6-luna", temperature: float = 0.0) -> ChatOpenAI:
    """
    Creates a OpenAI LLM.

    Args:
        model_name (str): The model to use. Default is "gpt-5.6-luna".
        temperature (float): The temperature to use.
    
    Returns:
        ChatOpenAI: The OpenAI LLM.
    """
    # Initialize the OpenAI language model with specified parameters
    llm = ChatOpenAI(
        model=model_name,  # Specify which OpenAI model to use
        # temperature=temperature,  # Controls randomness in responses (0.0 = deterministic)
        api_key=os.getenv("OPENAI_API_KEY"),
        reasoning_effort="none"  # Disable reasoning
    )
    
    return llm

# Create an instance of the OpenAI LLM using default parameters
llm = create_openai_llm()

# Send a test message to the LLM and print its response
# invoke() sends the message and returns an AIMessage object
# .content gets the actual response text
print(llm.invoke("Hello, how are you?").content)

print('################')

# Initialize chat history with first human message
messages = [HumanMessage(content="Hello, how are you?")]

# Send messages to LLM and get response
# llm.invoke(messages) will return a AIMessage object and AIMessage object has a variable called content
response = llm.invoke(messages)

# Add AI's response to chat history
messages.append(response)

# Add follow-up question to chat history
# This tests if the LLM can remember previous messages
messages.append(HumanMessage(content="What is the first thing I asked?"))

# Get response from LLM with full chat history context
response = llm.invoke(messages)

# Print just the content of the AI's response
print(response.content)
