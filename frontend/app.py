import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = 30


def error_detail(response: requests.Response) -> str:
    try:
        return str(response.json().get("detail", response.text))
    except ValueError:
        return response.text or f"Request failed with status {response.status_code}."


def request_json(method: str, path: str, **kwargs) -> dict:
    response = requests.request(
        method,
        f"{API_BASE_URL}{path}",
        timeout=kwargs.pop("timeout", REQUEST_TIMEOUT),
        **kwargs,
    )
    if not response.ok:
        raise RuntimeError(error_detail(response))
    return response.json()


st.set_page_config(
    page_title="Lexi-RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Lexi-RAG: Domain-Specific Assistant")
st.markdown(
    "Chat normally, then add PDFs to a conversation whenever you need "
    "document-grounded answers. Your chats and messages are saved locally."
)

try:
    chats = request_json("GET", "/chats")["chats"]
    if not chats:
        created = request_json("POST", "/chats", json={"title": "New chat"})
        chats = [created["chat"]]
except (requests.RequestException, RuntimeError, KeyError) as exc:
    st.error(f"Could not load saved chats. Is the FastAPI backend running? {exc}")
    st.stop()

chat_ids = [chat["id"] for chat in chats]
chat_by_id = {chat["id"]: chat for chat in chats}
if st.session_state.get("selected_chat_id") not in chat_by_id:
    st.session_state.selected_chat_id = chat_ids[0]

with st.sidebar:
    st.header("Chats")
    if st.button("＋ New chat", use_container_width=True):
        try:
            created = request_json("POST", "/chats", json={"title": "New chat"})
            st.session_state.selected_chat_id = created["chat"]["id"]
            st.rerun()
        except (requests.RequestException, RuntimeError, KeyError) as exc:
            st.error(f"Could not create chat: {exc}")

    selected_chat_id = st.selectbox(
        "Saved conversations",
        options=chat_ids,
        format_func=lambda chat_id: chat_by_id[chat_id]["title"],
        key="selected_chat_id",
    )
    selected_chat = chat_by_id[selected_chat_id]

    with st.expander("Chat settings"):
        new_title = st.text_input(
            "Chat title",
            value=selected_chat["title"],
            key=f"title_{selected_chat_id}",
            max_chars=100,
        )
        if st.button("Rename chat", key=f"rename_{selected_chat_id}"):
            try:
                request_json(
                    "PATCH",
                    f"/chats/{selected_chat_id}",
                    json={"title": new_title},
                )
                st.rerun()
            except (requests.RequestException, RuntimeError) as exc:
                st.error(f"Could not rename chat: {exc}")

        confirm_delete = st.checkbox(
            "Confirm deleting this chat and its documents",
            key=f"confirm_delete_{selected_chat_id}",
        )
        if st.button(
            "Delete chat",
            key=f"delete_{selected_chat_id}",
            disabled=not confirm_delete,
        ):
            try:
                request_json("DELETE", f"/chats/{selected_chat_id}")
                st.rerun()
            except (requests.RequestException, RuntimeError) as exc:
                st.error(f"Could not delete chat: {exc}")

    st.divider()
    st.subheader("Add context to this chat")
    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        key=f"upload_{selected_chat_id}",
    )
    if st.button("Add PDF to chat", disabled=uploaded_file is None):
        if uploaded_file is not None:
            with st.spinner("Processing PDF and creating embeddings..."):
                try:
                    request_json(
                        "POST",
                        f"/chats/{selected_chat_id}/documents",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                "application/pdf",
                            )
                        },
                        timeout=300,
                    )
                    st.success("PDF added. Your existing conversation was preserved.")
                except (requests.RequestException, RuntimeError) as exc:
                    st.error(f"Could not process PDF: {exc}")

    st.caption("PDFs are only used in the conversation they were added to.")

try:
    chat_data = request_json("GET", f"/chats/{selected_chat_id}")
except (requests.RequestException, RuntimeError, KeyError) as exc:
    st.error(f"Could not load this conversation: {exc}")
    st.stop()

documents = chat_data["documents"]
if documents:
    st.caption(
        "Attached documents: "
        + ", ".join(
            f"{document['filename']} ({document['status']})"
            for document in documents
        )
    )
else:
    st.caption("No documents attached. You can chat normally or add a PDF in the sidebar.")

for message in chat_data["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["sources"]:
            source_labels = [
                f"{source['filename']} (page {source['page']})"
                for source in message["sources"]
            ]
            st.caption("Sources: " + ", ".join(source_labels))

if prompt := st.chat_input("Message Lexi-RAG..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = request_json(
                    "POST",
                    f"/chats/{selected_chat_id}/messages",
                    json={"content": prompt},
                    timeout=180,
                )
                st.markdown(result["answer"])
                if result["sources"]:
                    source_labels = [
                        f"{source['filename']} (page {source['page']})"
                        for source in result["sources"]
                    ]
                    st.caption("Sources: " + ", ".join(source_labels))
                st.rerun()
            except (requests.RequestException, RuntimeError, KeyError) as exc:
                st.error(f"Error generating response: {exc}")
