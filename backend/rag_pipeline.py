from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

CHROMA_PATH = Path(__file__).parent / "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-20b"
DOCUMENT_UNAVAILABLE_MESSAGE = (
    "I am sorry, this information is not available in the provided document."
)


@lru_cache(maxsize=1)
def _embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _vector_store(include_embeddings: bool = True) -> Chroma:
    return Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=_embeddings() if include_embeddings else None,
    )


def process_pdf_and_create_vector_db(
    pdf_file_path: str,
    chat_id: str,
    document_id: str,
    filename: str,
) -> None:
    loader = PyPDFLoader(pdf_file_path)
    documents = loader.load()
    if not documents:
        raise ValueError("The PDF does not contain any readable pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents)
    if not chunks:
        raise ValueError("The PDF does not contain any readable text.")

    for chunk in chunks:
        chunk.metadata.update(
            {
                "chat_id": chat_id,
                "document_id": document_id,
                "source_name": filename,
            }
        )

    _vector_store().add_documents(chunks)


def delete_document_vectors(chat_id: str, document_id: str) -> None:
    _vector_store(include_embeddings=False).delete(
        where={"$and": [{"chat_id": chat_id}, {"document_id": document_id}]}
    )


def delete_chat_vectors(chat_id: str) -> None:
    _vector_store(include_embeddings=False).delete(where={"chat_id": chat_id})


def _source_for_document(document: Any) -> dict[str, int | str]:
    page = document.metadata.get("page", "Unknown")
    if isinstance(page, int):
        page += 1
    return {
        "filename": str(document.metadata.get("source_name", "Document")),
        "page": page,
    }


def ask_question(
    query: str,
    chat_id: str,
    history: list[dict[str, str]],
    has_documents: bool,
) -> dict[str, Any]:
    retrieved_documents = []
    if has_documents:
        retrieved_documents = _vector_store().similarity_search(
            query,
            k=3,
            filter={"chat_id": chat_id},
        )

    if retrieved_documents:
        context = "\n\n".join(
            (
                f"[Source: {document.metadata.get('source_name', 'Document')}, "
                f"page {_source_for_document(document)['page']}]\n"
                f"{document.page_content}"
            )
            for document in retrieved_documents
        )
        system_prompt = (
            "Answer the user using only the supplied document context. "
            f"If the context does not contain the answer, reply exactly: "
            f"'{DOCUMENT_UNAVAILABLE_MESSAGE}' Do not invent document facts. "
            "Use the source labels in the context when citing information.\n\n"
            f"Document context:\n{context}"
        )
    elif has_documents:
        system_prompt = (
            "The chat has uploaded documents, but no relevant passages were found "
            "for this question. Do not invent document facts. Reply exactly: "
            f"'{DOCUMENT_UNAVAILABLE_MESSAGE}'"
        )
    else:
        system_prompt = (
            "You are a helpful assistant. Answer the user's question clearly and "
            "use the conversation history when relevant."
        )

    messages = [SystemMessage(content=system_prompt)]
    for message in history:
        if message["role"] == "user":
            messages.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            messages.append(AIMessage(content=message["content"]))
    messages.append(HumanMessage(content=query))

    response = ChatGroq(model_name=LLM_MODEL, temperature=0.1).invoke(messages)
    answer = response.content
    if not isinstance(answer, str):
        answer = str(answer)

    sources: list[dict[str, int | str]] = []
    seen_sources: set[tuple[str, int | str]] = set()
    for document in retrieved_documents:
        source = _source_for_document(document)
        source_key = (str(source["filename"]), source["page"])
        if source_key not in seen_sources:
            sources.append(source)
            seen_sources.add(source_key)

    return {"answer": answer, "sources": sources}
