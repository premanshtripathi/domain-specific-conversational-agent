from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import shutil

# Importing the logic from our rag_pipeline module
from rag_pipeline import process_pdf_and_create_vector_db, ask_question

app = FastAPI(
    title="Lexi-RAG API",
    description="Backend API for the RAG-based Document Assistant",
    version="1.0.0"
)

# Directory to temporarily store uploaded PDFs
UPLOAD_DIR = "../data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    query: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Endpoint to handle PDF uploads, save them locally, 
    and trigger the vector database creation process.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # Save the uploaded file to the local directory
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Process the saved PDF and generate embeddings
        process_pdf_and_create_vector_db(file_path)
        return {
            "status": "success",
            "message": "Document processed and knowledge base updated successfully.",
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/chat")
async def chat(request: QueryRequest):
    """
    Endpoint to receive user queries, fetch context from the vector database, 
    and return the LLM's response along with source page numbers.
    """
    try:
        # Fetch answer and sources using the RAG pipeline
        response_data = ask_question(request.query)
        return {
            "status": "success",
            "answer": response_data["answer"],
            "sources": response_data["sources"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

# To run this server, use the following command in the terminal:
# uvicorn main:app --reload