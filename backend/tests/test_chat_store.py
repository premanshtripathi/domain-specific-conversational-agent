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
        self.assertEqual(self.store.ready_document_ids(chat["id"]), ["doc-1"])

        self.assertTrue(self.store.delete_chat(chat["id"]))
        self.assertIsNone(self.store.get_chat(chat["id"]))

    def test_message_updates_chat_ordering_and_only_titles_new_chats(self):
        chat = self.store.create_chat()
        self.store.add_message(chat["id"], "user", "First question")
        self.store.rename_chat(chat["id"], "Custom title")
        self.store.add_message(chat["id"], "user", "Another question")

        self.assertEqual(self.store.get_chat(chat["id"])["title"], "Custom title")
        self.assertEqual(self.store.list_chats()[0]["id"], chat["id"])

    def test_search_matches_chat_title_and_message_content(self):
        first_chat = self.store.create_chat("Biology notes")
        second_chat = self.store.create_chat("General questions")
        self.store.add_message(second_chat["id"], "user", "Explain the mitochondria")

        self.assertEqual(
            [chat["id"] for chat in self.store.list_chats("biology")],
            [first_chat["id"]],
        )
        self.assertEqual(
            [chat["id"] for chat in self.store.list_chats("MITOCHONDRIA")],
            [second_chat["id"]],
        )

    def test_history_is_limited_by_message_count_and_character_budget(self):
        chat = self.store.create_chat()
        self.store.add_message(chat["id"], "user", "A" * 8)
        self.store.add_message(chat["id"], "assistant", "B" * 8)
        self.store.add_message(chat["id"], "user", "C" * 8)

        history = self.store.recent_messages(chat["id"], limit=10, max_chars=16)

        self.assertEqual(history, [
            {"role": "assistant", "content": "B" * 8},
            {"role": "user", "content": "C" * 8},
        ])

    def test_ready_document_ids_are_scoped_and_deleted_with_document(self):
        first_chat = self.store.create_chat()
        second_chat = self.store.create_chat()
        self.store.add_document(first_chat["id"], "ready", "a.pdf", "/tmp/a.pdf")
        self.store.set_document_status("ready", "ready")
        self.store.add_document(first_chat["id"], "failed", "b.pdf", "/tmp/b.pdf")
        self.store.set_document_status("failed", "failed")
        self.store.add_document(second_chat["id"], "other", "c.pdf", "/tmp/c.pdf")
        self.store.set_document_status("other", "ready")

        self.assertEqual(self.store.ready_document_ids(first_chat["id"]), ["ready"])
        self.assertTrue(self.store.delete_document(first_chat["id"], "ready"))
        self.assertEqual(self.store.ready_document_ids(first_chat["id"]), [])
        self.assertFalse(self.store.delete_document(first_chat["id"], "other"))


if __name__ == "__main__":
    unittest.main()
