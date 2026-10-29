import tempfile
import unittest
from pathlib import Path

from chat_store import ChatStore


class ChatStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "chats.sqlite3"
        self.store = ChatStore(self.database_path)
        self.store.initialize()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_chat_and_messages_persist_across_store_instances(self):
        chat = self.store.create_chat()
        self.store.add_message(chat["id"], "user", "  Explain this topic  ")
        self.store.add_message(
            chat["id"],
            "assistant",
            "Here is an explanation.",
            [{"filename": "notes.pdf", "page": 2}],
        )

        reopened_store = ChatStore(self.database_path)
        reopened_store.initialize()
        loaded_chat = reopened_store.get_chat(chat["id"])
        messages = reopened_store.list_messages(chat["id"])

        self.assertEqual(loaded_chat["title"], "Explain this topic")
        self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
        self.assertEqual(messages[1]["sources"], [{"filename": "notes.pdf", "page": 2}])

    def test_rename_and_delete_chat(self):
        chat = self.store.create_chat()
        self.store.add_document(chat["id"], "doc-1", "notes.pdf", "/tmp/notes.pdf")
        self.store.set_document_status("doc-1", "ready")

        self.assertTrue(self.store.rename_chat(chat["id"], "Study notes"))
        self.assertEqual(self.store.get_chat(chat["id"])["title"], "Study notes")
        self.assertEqual(self.store.ready_document_count(chat["id"]), 1)

        self.assertTrue(self.store.delete_chat(chat["id"]))
        self.assertIsNone(self.store.get_chat(chat["id"]))

    def test_message_updates_chat_ordering_and_only_titles_new_chats(self):
        chat = self.store.create_chat()
        self.store.add_message(chat["id"], "user", "First question")
        self.store.rename_chat(chat["id"], "Custom title")
        self.store.add_message(chat["id"], "user", "Another question")

        self.assertEqual(self.store.get_chat(chat["id"])["title"], "Custom title")
        self.assertEqual(self.store.list_chats()[0]["id"], chat["id"])


if __name__ == "__main__":
    unittest.main()
