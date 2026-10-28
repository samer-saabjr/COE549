# Import required modules
from langchain_chroma import Chroma  # Vector database
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # Google's LLM and embeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage  # Message types for chat
from langgraph.prebuilt import create_react_agent  # For creating an agent that can use tools
from langchain.tools import tool  # Decorator to create tools

import os
from dotenv import load_dotenv

load_dotenv()

# API key for Google services
# Tool definition for adding numbers
@tool
def add_two_numbers(a: int, b: int) -> str:
    """
    Adds two numbers together
    Args:
        a (int): The first number
        b (int): The second number
    Returns:
        str: The sum of the two numbers
    """
    # Convert result to string since LLM expects string output
    return str(a + b)

# Tool definition for searching vector database
@tool
def search_vector_db(query: str) -> str:
    """
    Search the vector database for documents similar to the query.
    Args:
        query (str): The search query string to find relevant documents
    Returns:
        str: A concatenated string of the top 5 most similar document contents found in the vector database
    """
    # Initialize embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
    
    # Initialize/connect to vector database
    vector_store = Chroma(
        collection_name = "embeddings",
        embedding_function = embeddings,
        persist_directory = "./vector_db",
        collection_metadata = {"hnsw:space": "cosine"}  # Use cosine similarity
    )

    # Debug print
    print("Searching the vector database for: ", query)
    
    # Perform similarity search and get top 5 results
    result = vector_store.similarity_search(query = query, k = 5)
    # Combine all document contents into single string
    result_str = "\n".join([doc.page_content for doc in result])
    
    return result_str

# Main chat class that uses OpenAI LLM
class OpenAIChat:
    def __init__(self, model_name: str = "gpt-5.6-luna", temperature: float = 0.0):
        """
        Initialize OpenAIChat with a language model.

        Args:
            model_name (str): The model to use. Default is "gpt-5.6-luna".
            temperature (float): The temperature to use. Default is 0.0.
        """
        # Store API key
        
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            reasoning_effort="none"  # Disable reasoning
        )
        
        # Create agent with both tools available
        self.agent = create_react_agent(self.llm, tools=[add_two_numbers, search_vector_db])
        
        # # Initialize conversation history
        # self.messages = []

        # Initialize conversation with a system message that prompts/instructs the LLM on its role and capabilities
        # This is an example of prompt engineering - giving the LLM clear instructions on how to behave and what tools it can use
        self.messages = [HumanMessage(content="You are a helpful AI assistant that can search through documents and use the appropriate tool when needed to help answer questions. When you use a tool you will strictly adhere to the tool output and not use additional information. You will be polite.")]
        
    def send_message(self, message: str) -> str:
        """
        Send a message and get response from the model.
        
        Args:
            message (str): The message to send
            
        Returns:
            str: The model's response content
        """
        # Add user message to history
        self.messages.append(HumanMessage(content=message))
        
        # Store current history length to identify new messages later
        # We need this because the agent returns the full conversation history including old messages,
        # but we only want to return the new messages from this specific interaction
        history_length = len(self.messages)
        
        # Get response from agent, including any tool usage
        self.messages = self.agent.invoke({"messages": self.messages})["messages"]
        
        # Extract only the new messages from this interaction
        new_messages = self.messages[history_length:]

        return new_messages 

# Create chat instance
chat = OpenAIChat()

# Example query using vector database search
messages = chat.send_message("Give me examples on language ambiguity based on the documents in the vector database")

# Print all messages, handling both regular messages and tool calls
for message in messages:
    if message.content:
        print(type(message), message.content)
    else:
        print(type(message), message.tool_calls[0])
