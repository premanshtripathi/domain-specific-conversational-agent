import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


DATABASE_PATH = Path(__file__).parent / "chat_data.sqlite3"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatStore:
    def __init__(self, database_path: Path | str = DATABASE_PATH):
        self.database_path = Path(database_path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    sources TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS messages_chat_id_id
                    ON messages(chat_id, id);
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                    filename TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('processing', 'ready', 'failed')),
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS documents_chat_id
                    ON documents(chat_id, created_at);
                """
            )

    def create_chat(self, title: str = "New chat") -> dict[str, Any]:
        chat_id = str(uuid4())
        now = _now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO chats (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (chat_id, title, now, now),
            )
        chat = self.get_chat(chat_id)
        if chat is None:
            raise RuntimeError("The newly created chat could not be loaded.")
        return chat

    def list_chats(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT c.id, c.title, c.created_at, c.updated_at,
                       (SELECT COUNT(*) FROM messages m WHERE m.chat_id = c.id) AS message_count
                FROM chats c
                ORDER BY c.updated_at DESC, c.id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_chat(self, chat_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, title, created_at, updated_at FROM chats WHERE id = ?",
                (chat_id,),
            ).fetchone()
        return dict(row) if row else None

    def rename_chat(self, chat_id: str, title: str) -> bool:
        with self._connect() as connection:
            result = connection.execute(
                "UPDATE chats SET title = ?, updated_at = ? WHERE id = ?",
                (title, _now(), chat_id),
            )
        return result.rowcount > 0

    def add_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        sources: list[dict[str, int | str]] | None = None,
    ) -> int:
        now = _now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO messages (chat_id, role, content, sources, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (chat_id, role, content, json.dumps(sources or []), now),
            )
            if role == "user":
                title = " ".join(content.split())[:50] or "New chat"
                connection.execute(
                    """
                    UPDATE chats
                    SET title = CASE WHEN title = 'New chat' THEN ? ELSE title END,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (title, now, chat_id),
                )
            else:
                connection.execute(
                    "UPDATE chats SET updated_at = ? WHERE id = ?",
                    (now, chat_id),
                )
        return int(cursor.lastrowid)

    def list_messages(self, chat_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, role, content, sources, created_at
                FROM messages WHERE chat_id = ? ORDER BY id
                """,
                (chat_id,),
            ).fetchall()
        messages = []
        for row in rows:
            message = dict(row)
            message["sources"] = json.loads(message["sources"])
            del message["id"]
            messages.append(message)
        return messages

    def recent_messages(self, chat_id: str, limit: int = 20) -> list[dict[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content FROM messages
                WHERE chat_id = ? ORDER BY id DESC LIMIT ?
                """,
                (chat_id, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def add_document(
        self, chat_id: str, document_id: str, filename: str, stored_path: str
    ) -> None:
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (id, chat_id, filename, stored_path, status, created_at)
                VALUES (?, ?, ?, ?, 'processing', ?)
                """,
                (document_id, chat_id, filename, stored_path, now),
            )
            connection.execute(
                "UPDATE chats SET updated_at = ? WHERE id = ?",
                (now, chat_id),
            )

    def set_document_status(self, document_id: str, status: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE documents SET status = ? WHERE id = ?",
                (status, document_id),
            )

    def list_documents(self, chat_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, filename, status, created_at
                FROM documents WHERE chat_id = ? ORDER BY created_at, id
                """,
                (chat_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def ready_document_count(self, chat_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM documents WHERE chat_id = ? AND status = 'ready'",
                (chat_id,),
            ).fetchone()
        return int(row[0])

    def document_paths(self, chat_id: str) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT stored_path FROM documents WHERE chat_id = ?",
                (chat_id,),
            ).fetchall()
        return [str(row["stored_path"]) for row in rows]

    def delete_chat(self, chat_id: str) -> bool:
        with self._connect() as connection:
            result = connection.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        return result.rowcount > 0
