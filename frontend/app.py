import streamlit as st
import requests

# Constants
API_BASE_URL = "http://127.0.0.1:8000"

# Set up the Streamlit page configuration
st.set_page_config(
    page_title="Lexi-RAG Assistant",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for a cleaner, academic look
st.markdown("""
<style>
    .stChatFloatingInputContainer {
        padding-bottom: 20px;
    }
    .source-box {
        font-size: 0.8em;
        color: #4CAF50;
        background-color: #e8f5e9;
        padding: 4px 8px;
        border-radius: 4px;
        margin-top: 8px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

st.title("📚 Lexi-RAG: Domain-Specific Document Assistant")
st.markdown("Upload a PDF document to provide context, and then ask questions. The assistant will answer strictly based on the uploaded document and cite the page numbers.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for file upload
with st.sidebar:
    st.header("1. Upload Knowledge Base")
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if st.button("Process Document"):
        if uploaded_file is not None:
            with st.spinner("Processing document and generating embeddings... This may take a moment."):
                try:
                    # Send the file to the FastAPI backend
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(f"{API_BASE_URL}/upload", files=files)
                    
                    if response.status_code == 200:
                        st.success("Document processed successfully! You can now ask questions.")
                        # Clear chat history when a new document is uploaded
                        st.session_state.messages = []
                    else:
                        st.error(f"Error processing document: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Failed to connect to the backend server. Is FastAPI running? Error: {str(e)}")
        else:
            st.warning("Please upload a PDF file first.")
            
    st.markdown("---")
    st.subheader("System Architecture")
    st.markdown("- **UI:** Streamlit\n- **Backend:** FastAPI\n- **RAG Engine:** LangChain\n- **Vector DB:** Chroma\n- **LLM:** Llama-3 (Groq API)")

# Display chat history
st.header("2. Chat with Document")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            sources_str = ", ".join([str(s) for s in message["sources"]])
            st.markdown(f'<div class="source-box">Sources: Page(s) {sources_str}</div>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask a question about the uploaded document..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get bot response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Searching document and generating answer..."):
            try:
                # Send the query to the FastAPI backend
                payload = {"query": prompt}
                response = requests.post(f"{API_BASE_URL}/chat", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer received.")
                    sources = data.get("sources", [])
                    
                    # Display the answer
                    message_placeholder.markdown(answer)
                    
                    # Display sources if available
                    if sources:
                        sources_str = ", ".join([str(s) for s in sources])
                        st.markdown(f'<div class="source-box">Sources: Page(s) {sources_str}</div>', unsafe_allow_html=True)
                        
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                else:
                    error_msg = f"Error: {response.json().get('detail', 'Unknown error')}"
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
            except Exception as e:
                error_msg = f"Failed to connect to the backend server. Error: {str(e)}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})