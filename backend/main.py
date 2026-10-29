import logging
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from chat_store import ChatStore
from rag_pipeline import (
    delete_chat_vectors,
    delete_document_vectors,
    process_pdf_and_create_vector_db,
    ask_question,
)


logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024
UPLOAD_CHUNK_SIZE = 1024 * 1024
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
store = ChatStore()
store.initialize()

app = FastAPI(
    title="Lexi-RAG API",
    description="Backend API for the RAG-based Document Assistant",
    version="2.0.0",
)


class ChatCreateRequest(BaseModel):
    title: str = Field(default="New chat", min_length=1, max_length=100)


class ChatRenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=100)


class ChatMessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=10000)


@app.get("/chats")
def list_chats(search: str = Query(default="", max_length=100)):
    return {"chats": store.list_chats(search.strip())}


@app.post("/chats", status_code=201)
def create_chat(request: ChatCreateRequest):
    title = request.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="Chat title cannot be blank.")
    return {"chat": store.create_chat(title)}


@app.get("/chats/{chat_id}")
def get_chat(chat_id: str):
    chat = store.get_chat(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found.")
    return {
        "chat": chat,
        "messages": store.list_messages(chat_id),
        "documents": store.list_documents(chat_id),
    }


@app.patch("/chats/{chat_id}")
def rename_chat(chat_id: str, request: ChatRenameRequest):
    title = request.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="Chat title cannot be blank.")
    if not store.rename_chat(chat_id, title):
        raise HTTPException(status_code=404, detail="Chat not found.")
    return {"chat": store.get_chat(chat_id)}


@app.delete("/chats/{chat_id}")
def delete_chat(chat_id: str):
    if store.get_chat(chat_id) is None:
        raise HTTPException(status_code=404, detail="Chat not found.")
    stored_paths = store.document_paths(chat_id)
    if stored_paths:
        try:
            delete_chat_vectors(chat_id)
        except Exception as exc:
            logger.exception("Could not remove vectors for chat %s", chat_id)
            raise HTTPException(
                status_code=500, detail=f"Could not delete chat documents: {exc}"
            ) from exc

    for stored_path in stored_paths:
        try:
            Path(stored_path).unlink(missing_ok=True)
        except OSError as exc:
            logger.exception("Could not remove uploaded file %s", stored_path)
            raise HTTPException(
                status_code=500, detail=f"Could not remove uploaded file: {exc}"
            ) from exc
    store.delete_chat(chat_id)
    return {"status": "success"}


@app.post("/chats/{chat_id}/documents", status_code=201)
async def upload_document(chat_id: str, file: UploadFile = File(...)):
    if store.get_chat(chat_id) is None:
        raise HTTPException(status_code=404, detail="Chat not found.")
    filename = file.filename or ""
    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    document_id = str(uuid4())
    file_path = UPLOAD_DIR / f"{document_id}.pdf"
    document_record_created = False
    vectors_may_exist = False
    try:
        header = bytearray()
        total_size = 0
        with file_path.open("wb") as saved_file:
            while chunk := await file.read(UPLOAD_CHUNK_SIZE):
                total_size += len(chunk)
                if total_size > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="PDF exceeds the 20 MiB upload limit.",
                    )
                if len(header) < 1024:
                    header.extend(chunk[: 1024 - len(header)])
                saved_file.write(chunk)

        if b"%PDF-" not in header:
            raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF.")
        store.add_document(chat_id, document_id, Path(filename).name, str(file_path))
        document_record_created = True
        vectors_may_exist = True
        process_pdf_and_create_vector_db(str(file_path), chat_id, document_id, filename)
        store.set_document_status(document_id, "ready")
    except HTTPException:
        file_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        logger.exception("Could not process uploaded document %s", filename)
        if vectors_may_exist:
            try:
                delete_document_vectors(chat_id, document_id)
            except Exception:
                logger.exception("Could not roll back vectors for document %s", document_id)
        file_path.unlink(missing_ok=True)
        if document_record_created:
            try:
                store.set_document_status(document_id, "failed")
            except Exception:
                logger.exception("Could not mark document %s as failed", document_id)
        raise HTTPException(
            status_code=500, detail=f"Error processing document: {exc}"
        ) from exc
    finally:
        await file.close()

    return {
        "status": "success",
        "message": "Document processed successfully.",
        "document": {
            "id": document_id,
            "filename": Path(filename).name,
            "status": "ready",
        },
    }


@app.delete("/chats/{chat_id}/documents/{document_id}")
def delete_document(chat_id: str, document_id: str):
    if store.get_chat(chat_id) is None:
        raise HTTPException(status_code=404, detail="Chat not found.")
    document = store.get_document(chat_id, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found in this chat.")
    if document["status"] == "processing":
        raise HTTPException(
            status_code=409,
            detail="This document is still processing. Try again when processing completes.",
        )

    try:
        delete_document_vectors(chat_id, document_id)
        Path(document["stored_path"]).unlink(missing_ok=True)
        store.delete_document(chat_id, document_id)
    except Exception as exc:
        logger.exception("Could not remove document %s from chat %s", document_id, chat_id)
        raise HTTPException(
            status_code=500, detail=f"Could not remove document: {exc}"
        ) from exc
    return {"status": "success"}


@app.post("/chats/{chat_id}/messages")
def send_message(chat_id: str, request: ChatMessageRequest):
    if store.get_chat(chat_id) is None:
        raise HTTPException(status_code=404, detail="Chat not found.")
    content = request.content.strip()
    if not content:
        raise HTTPException(status_code=422, detail="Message cannot be blank.")

    history = store.recent_messages(chat_id)
    store.add_message(chat_id, "user", content)
    ready_document_ids = store.ready_document_ids(chat_id)
    try:
        response_data = ask_question(
            content,
            chat_id,
            history,
            has_documents=bool(ready_document_ids),
            document_ids=ready_document_ids,
        )
    except Exception as exc:
        logger.exception("Could not generate an answer for chat %s", chat_id)
        raise HTTPException(
            status_code=500, detail=f"Error generating response: {exc}"
        ) from exc

    store.add_message(
        chat_id,
        "assistant",
        response_data["answer"],
        response_data["sources"],
    )
    return {
        "status": "success",
        "answer": response_data["answer"],
        "sources": response_data["sources"],
    }
