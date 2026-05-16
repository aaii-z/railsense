import streamlit as st
import uuid

if "wkey" not in st.session_state:
    st.session_state.wkey = "init"

def cb():
    st.session_state.wkey = "chat_" + str(uuid.uuid4())
    st.session_state[st.session_state.wkey] = "Prefilled text!"

st.button("Trigger", on_click=cb)

val = st.chat_input("Input", key=st.session_state.wkey)
st.write("Current key:", st.session_state.wkey)
st.write("Widget value:", val)
