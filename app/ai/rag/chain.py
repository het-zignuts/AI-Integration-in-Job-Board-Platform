from app.ai.rag.prompt import PROMPT
from app.ai.llms.groq import get_groq_llm
from langchain_core.output_parsers import PydanticOutputParser
from app.ai.ai_schemas.response import AssistantResponse
import json

def generate_answer(context: str, question: str):
    """
    This functions creates a LangChain and returns the response genrated by passing the query along with the context to the chain.
    """
    if context is None:
        return "No matching information identified. Please upload relevant documents."

    llm=get_groq_llm() # get the LLM instance
    prompt=PROMPT # a reusable prompt template
    parser=PydanticOutputParser(pydantic_object=AssistantResponse) # parser instance to parser the LLM response accroding to given pydantic schema.

    chain= prompt | llm | parser # creating the LangChain

    response=chain.invoke({"context": context, "question": question}) # invoke the chain with the context and query values passed which will be substituted into the prompt before passing to LLM.
    return response.model_dump() # send a dict object as response