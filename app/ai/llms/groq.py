import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# loading the environment
load_dotenv()

def get_groq_llm():
    return ChatGroq(
        model=os.getenv("MODEL", "llama-3.1-8b-instant"),
        temperature=0.0,  
        api_key=os.getenv("GROQ_API_KEY")
    )