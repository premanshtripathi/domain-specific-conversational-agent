# Lexi-RAG

A local document assistant with persistent conversations, optional PDF context,
and source page references.

## Run locally

Activate the Conda environment used for this project, then install the backend
dependencies:

```bash
pip install -r backend/requirements.txt
```

Set `GROQ_API_KEY` in the root `.env` file. Start the API and Streamlit UI in
separate terminals:

```bash
cd backend
uvicorn main:app --reload
```

```bash
streamlit run frontend/app.py
```

The assistant supports ordinary conversations without a PDF. Uploading a PDF
attaches it to the currently selected chat without clearing its history. PDF
uploads are streamed to disk and limited to 20 MiB. A PDF must have a valid
`.pdf` filename and PDF signature. PDFs can be removed from a chat at any time.
Chat search matches titles and saved message text.
Conversations and messages are stored in `backend/chat_data.sqlite3`; uploaded
files and the Chroma vector database are stored under `backend/uploads/` and
`backend/chroma_db/`. These local data directories are ignored by Git.
Conversation history sent to the model is bounded to the latest 40 messages
and 16,000 characters; the full history remains saved locally.

## Chat API

- `GET /chats?search=term` — list saved chats, optionally matching chat titles
  and message contents.
- `POST /chats` — create a chat.
- `GET /chats/{chat_id}` — load a chat, its messages, and attached documents.
- `PATCH /chats/{chat_id}` — rename a chat.
- `DELETE /chats/{chat_id}` — remove a chat and its local documents.
- `DELETE /chats/{chat_id}/documents/{document_id}` — remove one attached PDF.
- `POST /chats/{chat_id}/messages` — send a message and save the response.
- `POST /chats/{chat_id}/documents` — add a PDF to a chat.

This first phase uses local, single-user storage and does not provide accounts
or synchronization between machines.