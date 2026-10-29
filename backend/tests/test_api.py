import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import main
from chat_store import ChatStore


class ChatApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        temporary_path = Path(self.temporary_directory.name)
        self.original_store = main.store
        self.original_upload_dir = main.UPLOAD_DIR
        main.store = ChatStore(temporary_path / "chats.sqlite3")
        main.store.initialize()
        main.UPLOAD_DIR = temporary_path / "uploads"
        main.UPLOAD_DIR.mkdir()
        self.client = TestClient(main.app)

    def tearDown(self):
        main.store = self.original_store
        main.UPLOAD_DIR = self.original_upload_dir
        self.temporary_directory.cleanup()

    @patch("main.process_pdf_and_create_vector_db")
    @patch("main.ask_question")
    def test_chat_history_is_saved_and_uploaded_pdf_does_not_clear_it(
        self, ask_question, process_pdf
    ):
        ask_question.return_value = {
            "answer": "A saved answer.",
            "sources": [],
        }
        created = self.client.post("/chats", json={"title": "New chat"})
        self.assertEqual(created.status_code, 201)
        chat_id = created.json()["chat"]["id"]

        answer = self.client.post(
            f"/chats/{chat_id}/messages",
            json={"content": "Can we chat without a PDF?"},
        )
        self.assertEqual(answer.status_code, 200)
        self.assertFalse(ask_question.call_args.kwargs["has_documents"])

        upload = self.client.post(
            f"/chats/{chat_id}/documents",
            files={"file": ("notes.pdf", b"%PDF-1.4\n", "application/pdf")},
        )
        self.assertEqual(upload.status_code, 201, upload.text)

        follow_up = self.client.post(
            f"/chats/{chat_id}/messages",
            json={"content": "Now use the uploaded notes."},
        )
        self.assertEqual(follow_up.status_code, 200)
        self.assertTrue(ask_question.call_args.kwargs["has_documents"])
        self.assertEqual(
            ask_question.call_args.args[2][-1],
            {"role": "assistant", "content": "A saved answer."},
        )

        detail = self.client.get(f"/chats/{chat_id}").json()
        self.assertEqual(
            [message["role"] for message in detail["messages"]],
            ["user", "assistant", "user", "assistant"],
        )
        self.assertEqual(detail["documents"][0]["filename"], "notes.pdf")
        self.assertEqual(ask_question.call_count, 2)
        process_pdf.assert_called_once()

    @patch("main.process_pdf_and_create_vector_db")
    @patch("main.delete_chat_vectors")
    def test_delete_chat_removes_its_uploaded_file_and_history(
        self, delete_vectors, process_pdf
    ):
        chat_id = self.client.post("/chats", json={}).json()["chat"]["id"]
        upload = self.client.post(
            f"/chats/{chat_id}/documents",
            files={"file": ("notes.pdf", b"%PDF-1.4\n", "application/pdf")},
        )
        self.assertEqual(upload.status_code, 201, upload.text)
        stored_file = main.UPLOAD_DIR / f"{upload.json()['document']['id']}.pdf"
        self.assertTrue(stored_file.exists())

        deleted = self.client.delete(f"/chats/{chat_id}")

        self.assertEqual(deleted.status_code, 200)
        self.assertFalse(stored_file.exists())
        self.assertEqual(self.client.get(f"/chats/{chat_id}").status_code, 404)
        delete_vectors.assert_called_once_with(chat_id)


if __name__ == "__main__":
    unittest.main()
