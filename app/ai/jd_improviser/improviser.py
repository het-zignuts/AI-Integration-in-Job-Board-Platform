from app.ai.embeddings.embed_utils import *
from app.models.job import Job
from app.ai.llms.groq import get_groq_llm
from app.ai.ai_schemas.response import *
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import os
from sqlmodel import Session, select, text

with open("prompt_improv.txt", "r") as file:
    template=file.read()

llm=get_groq_llm()
prompt=PromptTemplate(input_variables=["mode", "description"], template=template)
parser=PydanticOutputParser(pydantic_object=ImprovementResponse)
chain=prompt | llm  | parser

def get_improved_jd(mode: str, desc: str session: Session):
    llm_resp=chain.invoke({"mode": mode, "description": desc})
    return llm_resp
