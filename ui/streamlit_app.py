import streamlit as st

st.set_page_config(page_title="RailSense", page_icon="🚆")
st.title("🚆 RailSense")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("How can I help?")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": "placeholder response"})
    st.rerun()
