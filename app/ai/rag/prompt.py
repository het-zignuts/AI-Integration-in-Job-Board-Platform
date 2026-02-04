from langchain_core.prompts import PromptTemplate
import os
from pathlib import Path 

# Path to the reusable prompt template
BASE_DIR=Path(__file__).resolve().parent
file_pth=BASE_DIR/"template.txt"

# extract the prompt text from the file
with open(file_pth, "r") as file:
    template=file.read()

# prpare the langchin template with "context" and "question" marked as the input variables
PROMPT=PromptTemplate(input_variables=["context", "question"], template=template)