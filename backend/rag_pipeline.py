import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from the .env file
load_dotenv()

# Constants
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-20b"

def process_pdf_and_create_vector_db(pdf_file_path: str):
    """
    Step 1: Load the PDF, split it into chunks, and store vectors in ChromaDB.
    """
    print(f"Processing {pdf_file_path}...")
    
    # 1. Load PDF
    loader = PyPDFLoader(pdf_file_path)
    documents = loader.load()

    # 2. Split Text (chunk size of 1000, overlap of 200 for maintaining context)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # 3. Create Embeddings & Store in ChromaDB
    # Note: all-MiniLM-L6-v2 runs locally on CPU/GPU (no API cost)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Overwrites the existing DB so the new document is freshly indexed
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=CHROMA_PATH
    )
    print("Vector Database created successfully!")
    return True

def get_rag_chain():
    """
    Step 2: Set up the Retrieval and Generation chain.
    """
    # Setup Embeddings and Vector DB for retrieval
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # Fetches the top 3 most relevant chunks (k=3)
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Setup LLM via Groq (temperature set to 0.1 to remain factual and prevent hallucinations)
    llm = ChatGroq(model_name=LLM_MODEL, temperature=0.1)

    # Create Strict Prompt Template (This enforces 'Source Grounding' during evaluation)
    system_prompt = (
        "You are an intelligent assistant. Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer or if the answer is not present in the provided context, "
        "just say 'I am sorry, this information is not available in the provided document'. "
        "Do NOT hallucinate or make up information. Keep the answer concise and relevant.\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Build the final RAG pipeline
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

def ask_question(query: str):
    """
    Step 3: Process the user's query to generate an answer and extract source page numbers.
    """
    chain = get_rag_chain()
    response = chain.invoke({"input": query})
    
    answer = response["answer"]
    
    # Extract unique source page numbers
    sources = []
    for doc in response["context"]:
        # PyPDFLoader automatically adds the 'page' key to the metadata
        page = doc.metadata.get("page", "Unknown")
        # Page numbers are usually 0-indexed, so adding +1 for readability
        actual_page = page + 1 if isinstance(page, int) else page
        if actual_page not in sources:
            sources.append(actual_page)
            
    return {"answer": answer, "sources": sources}