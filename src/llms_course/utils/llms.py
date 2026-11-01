import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

load_dotenv()


def _get_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.template to .env and add your key."
        )
    return api_key


def get_llm(model: str = "gpt-5.6-luna", temperature: float | None = None):
    options = {"model": model, "api_key": _get_api_key()}
    if temperature is not None:
        options["temperature"] = temperature
    return ChatOpenAI(**options)


def get_structured_output_llm(
    schema: type[BaseModel],
    model: str = "gpt-5.6-luna",
    temperature: float | None = None,
):
    return get_llm(model=model, temperature=temperature).with_structured_output(schema)
