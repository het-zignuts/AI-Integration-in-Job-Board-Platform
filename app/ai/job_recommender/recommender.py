from app.ai.embeddings.embed_utils import *
from app.models.job import Job
from app.ai.llms.groq import get_groq_llm
from langchain_core.output_parsers import PydanticOutputParser
from app.ai.ai_schemas.response import *
from langchain_core.prompts import PromptTemplate
import os
from sqlmodel import Session, select, text

with open("prompt.txt", "r") as file:
    template=file.read()

llm=get_groq_llm()
parser=PydanticOutputParser(pydantic_object=AIRatingResponse)
prompt=PromptTemplate(input_variables=["context", "question"], template=template)
chain=prompt | llm | parser

question="Given the resume text and the job recommended based on that as context, please rate the job recommended with a confidence score and a shoe=rt 1-2 line reason."

def get_recs(resume_text: str, session: Session):
    embed_model=get_embed_model()
    embeddings=embed_model.embed_query(resume_text)
    vector_literal = f"ARRAY[{','.join(map(str, query_vector))}]::vector"
    query=f"""
    SELECT entity_type, entity_id, content
    FROM embeddings
    WHERE entity_type='job'
    ORDER BY embedding <#> {vector_literal}  -- pgvector cosine distance
    LIMIT :top_k
    """
    results=session.execute(text(query),{"query_vector": embeddings, "top_k": top_k}).all()
    matches=[]
    for entity_id, content in results:
        job=session.get(Job, entity_id)
        llm_resp=chain.invoke({"context": context, "question": question})
        conf_scr=llm_resp.confidence
        reason=llm_resp.reason
        match=JobMatch(job_id=job.id, job_title=job.title, match_reason=reason, confidence_score=conf_scr)
        matches.append(match)
    return matches

