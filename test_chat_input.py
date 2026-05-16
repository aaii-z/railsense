import streamlit as st

st.write("Hello")

if "my_chat_input" not in st.session_state:
    st.session_state.my_chat_input = ""

prompt = st.chat_input("Type or speak", key="my_chat_input", accept_audio=True)

if prompt:
    if getattr(prompt, "audio", None):
        st.write("Audio received!")
        st.session_state.my_chat_input = "This is a transcribed test!"
        st.rerun()
    elif getattr(prompt, "text", None):
        st.write("You sent text:", prompt.text)
