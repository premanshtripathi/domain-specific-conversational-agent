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
attaches it to the currently selected chat without clearing its history.
Conversations and messages are stored in `backend/chat_data.sqlite3`; uploaded
files and the Chroma vector database are stored under `backend/uploads/` and
`backend/chroma_db/`. These local data directories are ignored by Git.

## Chat API

- `GET /chats` — list saved chats.
- `POST /chats` — create a chat.
- `GET /chats/{chat_id}` — load a chat, its messages, and attached documents.
- `PATCH /chats/{chat_id}` — rename a chat.
- `DELETE /chats/{chat_id}` — remove a chat and its local documents.
- `POST /chats/{chat_id}/messages` — send a message and save the response.
- `POST /chats/{chat_id}/documents` — add a PDF to a chat.

This first phase uses local, single-user storage and does not provide accounts
or synchronization between machines.