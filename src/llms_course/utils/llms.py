from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

def get_llm(model = "gemini-2.5-flash", temperature = 0.0):
    llm = ChatGoogleGenerativeAI(model = model, temperature = temperature)
    return llm

def get_structured_output_llm(schema: BaseModel, model = "gemini-2.5-flash", temperature = 0.0):
    llm = ChatGoogleGenerativeAI(model = model, temperature = temperature)
    structured_output_llm = llm.with_structured_output(schema)
    return structured_output_llm