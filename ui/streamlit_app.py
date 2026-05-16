import streamlit as st
import sys
import uuid
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chatbot.dialogue import handle_message, init_dialogue_state
from voice.transcriber import transcribe

st.set_page_config(page_title="RailSense", page_icon="🚆")
st.title("🚆 RailSense")



if "messages" not in st.session_state:
    st.session_state.messages = []
if "dialogue_state" not in st.session_state:
    st.session_state.dialogue_state = init_dialogue_state()
if "text_to_process" not in st.session_state:
    st.session_state.text_to_process = None
if "chat_widget_key" not in st.session_state:
    st.session_state.chat_widget_key = "chat_widget_" + str(uuid.uuid4())

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
        if msg.get("timestamp"):
            st.caption(msg["timestamp"])

def handle_chat_submit():
    # The callback is triggered by the widget with the current key
    current_key = st.session_state.chat_widget_key
    val = st.session_state.get(current_key)
    
    if val:
        if getattr(val, "audio", None) is not None or (hasattr(val, "get") and val.get("audio")):
            audio_val = getattr(val, "audio", None) or val.get("audio")
            try:
                # Transcribe the audio
                text = transcribe(audio_val.getvalue())
                st.session_state.last_transcription = text  # Debug
                # Rotate the key to bypass Streamlit's trigger widget reset
                new_key = "chat_widget_" + str(uuid.uuid4())
                st.session_state.chat_widget_key = new_key
                # Set the text on the NEW key so it prefills
                st.session_state[new_key] = text
            except Exception as exc:
                st.error(f"Transcription failed: {exc}")
        elif getattr(val, "text", None) or (hasattr(val, "get") and val.get("text")):
            text_val = getattr(val, "text", None) or val.get("text")
            st.session_state.text_to_process = text_val
    elif isinstance(val, str) and val.strip():
        st.session_state.text_to_process = val

st.chat_input("How can I help?", key=st.session_state.chat_widget_key, accept_audio=True, on_submit=handle_chat_submit)

if st.session_state.get("last_transcription"):
    st.info(f"Debug: transcribed text was '{st.session_state.last_transcription}'")

# ── Process whichever input arrived ─────────────────────────────────────────
if st.session_state.text_to_process:
    text_to_process = st.session_state.text_to_process
    st.session_state.text_to_process = None

    user_ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.messages.append({
        "role": "user",
        "content": text_to_process,
        "timestamp": user_ts,
    })

    # Render user message immediately so it's visible during "Thinking..."
    with st.chat_message("user"):
        st.write(text_to_process)
        st.caption(user_ts)

    # Render assistant bubble with spinner inside it
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = handle_message(text_to_process, st.session_state.dialogue_state)
            except Exception as exc:
                response = {"message": f"Unexpected UI error while generating a reply: {exc}"}

        st.write(response["message"])
        journeys = response.get("journeys")
        prediction = response.get("prediction")
        debug = response.get("debug")
        if journeys:
            st.dataframe(journeys, use_container_width=True)
        if prediction:
            st.json(prediction)
        if debug:
            with st.expander("Debug details"):
                st.json(debug)
        bot_ts = datetime.now().strftime("%H:%M:%S")
        st.caption(bot_ts)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response["message"],
        "journeys": journeys,
        "prediction": prediction,
        "debug": debug,
        "timestamp": bot_ts,
    })
    st.rerun()
