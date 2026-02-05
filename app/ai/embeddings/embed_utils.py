from app.models.job import Job
from app.models.company import Company
from app.models.user import User
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

def get_embed_model():
    """Return a HuggingFace embedding model for semantic vectors."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_job_embedding_content(job: Job, company: Company) -> str:
     """Create a text representation of a job + company for embedding."""
    content=f"""
            Job Title: {job.title}
            Company: {company.name}
            Location: {job.location or "Not specified"}
            Mode: {job.mode.value}
            Employment Type: {job.employment_type.value}
            Salary: {job.remuneration_range or "Not specified"}
            Tags: {", ".join(job.tags)}
            Job Description: {job.description or ""}
        """
    return content.strip()

def build_company_embedding_content(company: Company) -> str:
    """Create a text representation of a company for embedding."""
    content=f"""
            Company Name: {company.name}
            Domain: {company.domain}
            Location: {company.location}
            Company Size: {company.company_size}
            Description: {company.description}
        """
    return content.strip()

def build_candidate_embedding_content(user: User, resume_txt) -> str:
    """Create a text representation of a candidate resume for embedding."""
    content=f"""
            Candidate Username: {user.user_name}
            Role: {user.role.value}
            Resume: {resume_text}
        """
    return content.strip()

def load_document(file_path: str) -> str:
    """
    This function loades the PDF file from the path specified.
    """
    path=Path(file_path)
    if not path.exists():
        raise FileNotFoundError("Document not found")
    if path.suffix.lower()==".txt":
        return path.read_text(encoding="utf-8") # return the read file content if it is a text file
    elif path.suffix.lower()==".pdf": # if the file is in PDF format
        loader=PyPDFLoader(str(path)) # create loader instance with the file path
        pages=loader.load() # load the file content
        text="\n\n".join(page.page_content for page in pages) # concatenate the content pages into a single text blob
        return text 
    else:
        raise ValueError("Unsupported file type. Only .txt and .pdf are supported.")
    


def chunk_text(text: str):
    """
    This function creates chunks from the text extracted from the file.
    """
    # Splitter instance
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=500, # generally, a couple of paragraphs for semantic completeness
        chunk_overlap=100, # ensures across-the-chunk-boundry info not lost
        separators=["\n\n", "\n", ".", " ", ""] # seperating characters for chunking.
    )
    return splitter.split_text(text)

def normalize_text(text: str) -> str:
    """
    This function normalizes the text before chunkig to ensure clean etx boundries. 
    Ensures no extra spaces, line breaks, extra numberings (page no.s, index, title, etc.)
    """
    text = re.sub(r"\n{2,}", "\n\n", text) # convert multiple line breaks to paragraph sepearation
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text) # replace single newline character with white space 
    text = re.sub(r"\s+", " ", text) # replace multiple spaces, tabs, etc. with single space.
    #trim trailing and leading whitespaces before returning as normalized text
    return text.strip()