import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

import rag_pipeline


class RagPipelineTests(unittest.TestCase):
    @patch("rag_pipeline._vector_store")
    @patch("rag_pipeline.ChatGroq")
    def test_normal_chat_uses_history_without_loading_document_store(
        self, chat_groq, vector_store
    ):
        model = chat_groq.return_value
        model.invoke.return_value.content = "A helpful answer."

        response = rag_pipeline.ask_question(
            "What did I ask earlier?",
            "chat-1",
            [
                {"role": "user", "content": "Remember this detail."},
                {"role": "assistant", "content": "I will."},
            ],
            has_documents=False,
        )

        sent_messages = model.invoke.call_args.args[0]
        self.assertEqual(response, {"answer": "A helpful answer.", "sources": []})
        self.assertEqual(sent_messages[1].content, "Remember this detail.")
        self.assertEqual(sent_messages[2].content, "I will.")
        self.assertEqual(sent_messages[3].content, "What did I ask earlier?")
        vector_store.assert_not_called()

    @patch("rag_pipeline._vector_store")
    @patch("rag_pipeline.ChatGroq")
    def test_document_search_is_scoped_to_chat_and_returns_named_sources(
        self, chat_groq, vector_store
    ):
        model = chat_groq.return_value
        model.invoke.return_value.content = "A document-grounded answer."
        vector_store.return_value.similarity_search.return_value = [
            Document(
                page_content="Important note.",
                metadata={
                    "chat_id": "chat-2",
                    "source_name": "lecture.pdf",
                    "page": 0,
                },
            )
        ]

        response = rag_pipeline.ask_question(
            "What does the lecture note say?",
            "chat-2",
            [],
            has_documents=True,
        )

        vector_store.return_value.similarity_search.assert_called_once_with(
            "What does the lecture note say?",
            k=3,
            filter={"chat_id": "chat-2"},
        )
        self.assertEqual(
            response["sources"],
            [{"filename": "lecture.pdf", "page": 1}],
        )
        self.assertIn("Important note.", chat_groq.return_value.invoke.call_args.args[0][0].content)


if __name__ == "__main__":
    unittest.main()
