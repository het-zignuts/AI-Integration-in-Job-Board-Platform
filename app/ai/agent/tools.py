from sqlmodel import Session, select, text
from app.ai.embeddings.embed_utils import get_embed_model
from app.ai.llms.groq import get_groq_llm
from app.ai.ai_schemas.response import *
import requests

llm=get_groq_llm() # LLM instance for reasoning and generation


def search_vector_db(session: Session, query: str, entity_type: str, top_k: int =5):
    """
    Perform a semantic search over the embeddings table using pgvector.
    Returns top_k closest matches for the given query and entity type.
    """
    embed_model=get_embed_model()
    embeddings=embed_model.embed_query(query) # convert query to vector
    vector_literal = f"ARRAY[{','.join(map(str, query_vector))}]::vector" # Convert Python list of floats into Postgres array literal

    # Query embeddings table ordered by cosine similarity
    stmt=f"""
    SELECT entity_type, entity_id, content
    FROM embeddings
    WHERE entity_type= :entity_type
    ORDER BY embedding <#> {vector_literal}  -- pgvector cosine distance
    LIMIT :top_k
    """
    results=session.execute(text(stmt),{"query_vector": embeddings, "entity_type": entity_type, "top_k": top_k}).all()
    return {
        "count": len(rows),
        "results": [
            {
                "entity_id": str(result.entity_id),
                "entity_type": result.entity_type,
                "content": result.content
            }
            for result in results
        ]
    }

def api_call(user, session, user_context: UserContext, action: ApiCallInput, action_method: str="GET"):
    """
    Executes an HTTP API call on behalf of the user.
    - Supports GET and POST requests.
    - Uses the user's access token for authentication.
    """
    headers={
        "Authorization": f"Bearer {user_context.access_token}"
    }
    BASE_URL="http://127.0.0.1:8000"
    endpoint=action.endpoint
    url=f"{BASE_URL}{endpoint}"
    if action_method=="GET":
        resp=requests.get(url, headers=headers)
    else:
        resp=requests.post(url, headers=headers)
    if resp.status_code>=400:
        raise RuntimeError(f"API call failed: {resp.text}")
    return resp.json()

def agent_llm(messages):
    """
    Sends a list of messages (system + user + assistant) to the LLM and returns the content.
    """
    return llm.invoke(messages).content

def llm_reasoning_tool(user_role, task, context, PROMPT):
    """
    Uses the LLM for reasoning tasks.
    - Combines user role, task, and previous context into a formatted prompt.
    - Returns LLM output as a string.
    """
    prompt=PROMPT.format(user_role=user_role, context=context, task=task)
    return llm.invoke(prompt).content
