# Import required modules from langchain
from langchain_openai import ChatOpenAI  # For interacting with Google's OpenAI model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage  # Message types for chat
from langgraph.prebuilt import create_react_agent # create_react_agent is a function that takes a language model and tools and returns an agent

import os
from dotenv import load_dotenv

load_dotenv()

class OpenAIChat:
    def __init__(self, model_name: str = "gpt-5.6-luna", temperature: float = 0.0):
        """
        Initialize OpenAIChat with a language model.

        Args:
            model_name (str): The model to use. Default is "gpt-5.6-luna".
            temperature (float): The temperature to use. Default is 0.0.
        """
        # Initialize the OpenAI language model with specified parameters
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            reasoning_effort="none"  # Disable reasoning
        )
        # Create a react agent that can use the LLM
        self.agent = create_react_agent(self.llm, tools=[])
        # Initialize empty list to store conversation history
        self.messages = []
        
    def send_message(self, message: str) -> str:
        """
        Send a message and get response from the model.
        
        Args:
            message (str): The message to send
            
        Returns:
            str: The model's response content
        """
        # Add user's message to conversation history
        self.messages.append(HumanMessage(content=message))
        # Get response from LLM using full conversation history
        response = self.llm.invoke(self.messages)
        # Add AI's response to conversation history
        self.messages.append(response)
        # Return just the content of the response
        return response.content

# Example usage
chat = OpenAIChat()  # Create new chat instance with default parameters
print(chat.send_message("Hello, how are you?"))  # Send initial greeting
print(chat.send_message("What is the first thing I asked?"))  # Test conversation memory
