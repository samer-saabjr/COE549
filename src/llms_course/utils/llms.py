from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import os
load_dotenv()

def get_llm(model = "gpt-5.6-luna", temperature = 0.0):
    llm = ChatOpenAI(model = model, api_key = os.getenv("OPENAI_API_KEY"))
    return llm

def get_structured_output_llm(schema: BaseModel, model = "gpt-5.6-luna", temperature = 0.0):
    llm = ChatOpenAI(model = model, api_key = os.getenv("OPENAI_API_KEY"))
    structured_output_llm = llm.with_structured_output(schema)
    return structured_output_llm