import streamlit as st

st.write("Hello")

def handle_chat():
    val = st.session_state.my_chat_input
    if isinstance(val, dict) and val.get("audio"):
        st.session_state.my_chat_input = "This is a transcribed test!"
        # In callback, modifying widget state usually works if we don't hit the "cannot modify after instantiation" because callback runs before the widget is instantiated in the current run.

if "my_chat_input" not in st.session_state:
    st.session_state.my_chat_input = ""

prompt = st.chat_input("Type or speak", key="my_chat_input", accept_audio=True, on_submit=handle_chat)

if prompt:
    if getattr(prompt, "audio", None):
        st.write("Audio received but shouldn't get here because callback modifies it!")
    elif getattr(prompt, "text", None):
        st.write("You sent text:", prompt.text)
