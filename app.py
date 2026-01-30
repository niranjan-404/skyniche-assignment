import streamlit as st
from processing import rag

st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("SKYNICHE Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask something from your document...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag(query, st.session_state.messages)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})