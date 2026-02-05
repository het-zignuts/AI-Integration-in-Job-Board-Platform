import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# loading the environment
load_dotenv()

def get_groq_llm():
    """
    Initialize and return a ChatGroq LLM instance using environment settings.
    """
    return ChatGroq(
        model=os.getenv("MODEL", "llama-3.1-8b-instant"), # default model
        temperature=0.0,  # deterministic responses
        api_key=os.getenv("GROQ_API_KEY")
    )