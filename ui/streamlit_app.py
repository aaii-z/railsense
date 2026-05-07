import streamlit as st
import sys
from pathlib import Path
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chatbot.dialogue import handle_message, init_dialogue_state

st.set_page_config(page_title="RailSense", page_icon="🚆")
st.title("🚆 RailSense")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "dialogue_state" not in st.session_state:
    st.session_state.dialogue_state = init_dialogue_state()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant":
            journeys = msg.get("journeys")
            prediction = msg.get("prediction")
            debug = msg.get("debug")
            if journeys:
                st.dataframe(journeys, use_container_width=True)
            if prediction:
                st.json(prediction)
            if debug:
                with st.expander("Debug details"):
                    st.json(debug)

user_input = st.chat_input("How can I help?")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    try:
        with st.spinner("Thinking..."):
            response = handle_message(user_input, st.session_state.dialogue_state)
    except Exception as exc:
        response = {
            "message": f"Unexpected UI error while generating a reply: {exc}",
        }
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response["message"],
            "journeys": response.get("journeys"),
            "prediction": response.get("prediction"),
            "debug": response.get("debug"),
        }
    )
    st.rerun()
