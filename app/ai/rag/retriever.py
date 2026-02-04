from sqlmodel import Session, text
from typing import List, Dict
from app.ai.embeddings.embed_utils import get_embed_model

def retrieve_context(session: Session, query: str, top_k: int = 5) -> str:
    embed_model=get_embed_model()
    query_vector=embed_model.embed_query(query)
    vector_literal = f"ARRAY[{','.join(map(str, query_vector))}]::vector"
    query=f"""
    SELECT entity_type, entity_id, content
    FROM embeddings
    ORDER BY embedding <#> {vector_literal}  -- pgvector cosine distance
    LIMIT :top_k
    """
    results=session.execute(text(query),{"query_vector": query_vector, "top_k": top_k}).all()
    retrieved_chunks=[f"""
                        entity_type: {r.entity_type}, 
                        entity_id: {r.entity_id},
                        content: {r.content}
                        """ 
                    for r in results]
    context="\n\n".join(retrieved_chunks)
    return context.strip()