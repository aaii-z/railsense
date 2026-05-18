import os
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

from chatbot.dialogue import handle_message, init_dialogue_state, GREETING, STAFF_GREETING
from db.conversations import get_sessions, get_session_messages
from voice.transcriber import transcribe

_STAFF_USERNAME = os.environ.get("STAFF_USERNAME", "staff")
_STAFF_PASSWORD = os.environ.get("STAFF_PASSWORD", "railsense123")

st.set_page_config(page_title="RailSense", page_icon="🚆", initial_sidebar_state="expanded")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": GREETING}]
if "dialogue_state" not in st.session_state:
    st.session_state.dialogue_state = init_dialogue_state()
if "is_staff" not in st.session_state:
    st.session_state.is_staff = False
if "page" not in st.session_state:
    st.session_state.page = "chat"
if "text_to_process" not in st.session_state:
    st.session_state.text_to_process = None
if "chat_widget_key" not in st.session_state:
    st.session_state.chat_widget_key = "chat_widget_" + str(uuid.uuid4())


# Sidebar: conversation history
if "expanded_sessions" not in st.session_state:
    st.session_state.expanded_sessions = set()

with st.sidebar:
    if st.button("+ New chat", key="sidebar_new_chat", use_container_width=True, type="primary"):
        st.session_state.session_id = str(uuid.uuid4())
        greeting = STAFF_GREETING if st.session_state.is_staff else GREETING
        st.session_state.messages = [{"role": "assistant", "content": greeting}]
        st.session_state.dialogue_state = init_dialogue_state()
        st.session_state.expanded_sessions.discard(st.session_state.session_id)
        st.rerun()

    st.divider()
    st.caption("Previous conversations")

    try:
        sessions = get_sessions()
    except Exception as exc:
        st.error(f"Couldn't load conversation history: {exc}")
        sessions = []

    current_sid = st.session_state.session_id
    if not any(s["session_id"] == current_sid for s in sessions):
        sessions = [{"session_id": current_sid, "label": "New conversation", "last_active": None}] + sessions

    if not sessions:
        st.caption("_No conversations yet._")

    for s in sessions:
        sid = s["session_id"]
        label = s["label"][:38] + "…" if len(s["label"]) > 38 else s["label"]
        is_active = sid == current_sid
        is_expanded = sid in st.session_state.expanded_sessions

        c1, c2 = st.columns([5, 1])
        with c1:
            btn_label = ("● " if is_active else "") + label
            if st.button(
                btn_label,
                key=f"sess_{sid}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                if not is_active:
                    msgs = get_session_messages(sid)
                    st.session_state.session_id = sid
                    st.session_state.messages = [
                        {
                            "role": m["role"],
                            "content": m["content"],
                            **(m.get("extras") or {}),
                        }
                        for m in msgs
                    ] or [{"role": "assistant", "content": STAFF_GREETING if st.session_state.is_staff else GREETING}]
                    st.session_state.dialogue_state = init_dialogue_state()
                    st.session_state.dialogue_state["history"] = [
                        {"role": m["role"], "content": m["content"]} for m in msgs
                    ][-20:]
                    st.rerun()
        with c2:
            arrow = "▲" if is_expanded else "▼"
            if st.button(arrow, key=f"toggle_{sid}"):
                if is_expanded:
                    st.session_state.expanded_sessions.discard(sid)
                else:
                    st.session_state.expanded_sessions.add(sid)
                st.rerun()

        if is_expanded:
            if is_active:
                preview_msgs = st.session_state.messages
            else:
                try:
                    preview_msgs = get_session_messages(sid)
                except Exception:
                    preview_msgs = []
            for m in preview_msgs:
                prefix = "**You:** " if m["role"] == "user" else "**Bot:** "
                snippet = m["content"][:80] + "…" if len(m["content"]) > 80 else m["content"]
                st.caption(prefix + snippet)
            st.markdown("---")


# Login page
if st.session_state.page == "login":
    if st.button("Back", key="login_back"):
        st.session_state.page = "chat"
        st.rerun()

    st.title("Staff Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Log in", key="login_submit", type="primary"):
        if username == _STAFF_USERNAME and password == _STAFF_PASSWORD:
            st.session_state.is_staff = True
            st.session_state.page = "chat"
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = [{"role": "assistant", "content": STAFF_GREETING}]
            st.session_state.dialogue_state = init_dialogue_state()
            st.rerun()
        else:
            st.error("Invalid credentials.")
    st.stop()


# Chat page
st.markdown("""
<style>
/* Don't hide the header or its toolbar, the sidebar toggle lives inside it.
   Just hide the noisy children we don't want. */
[data-testid="stDeployButton"],
[data-testid="stMainMenu"],
[data-testid="stStatusWidget"],
[data-testid="stDecoration"] { display: none !important; }

header[data-testid="stHeader"] {
    background: transparent !important;
    box-shadow: none !important;
    border-bottom: none !important;
}

/* Force the sidebar toggle visible. The data-testid changes between Streamlit
   versions so target all the names we've seen. */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999 !important;
}

.block-container { padding-top: 0.75rem !important; }
div[data-testid="stHorizontalBlock"]:first-of-type {
    align-items: center;
    border-bottom: 1px solid rgba(49, 51, 63, 0.15);
    padding-bottom: 0.5rem;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

col_title, col_auth = st.columns([5, 1])
with col_title:
    st.markdown("## 🚆 RailSense")
with col_auth:
    if st.session_state.is_staff:
        st.markdown(
            '<div style="background:#198754;color:white;padding:4px 10px;border-radius:16px;'
            'font-size:0.78rem;font-weight:600;text-align:center;margin-bottom:4px;">🔐 Staff</div>',
            unsafe_allow_html=True,
        )
        if st.button("Logout", key="chat_logout", use_container_width=True):
            st.session_state.is_staff = False
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = [{"role": "assistant", "content": GREETING}]
            st.session_state.dialogue_state = init_dialogue_state()
            st.rerun()
    else:
        if st.button("Staff Login", key="chat_login", use_container_width=True):
            st.session_state.page = "login"
            st.rerun()

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
    current_key = st.session_state.chat_widget_key
    val = st.session_state.get(current_key)

    if val:
        if getattr(val, "audio", None) is not None or (hasattr(val, "get") and val.get("audio")):
            audio_val = getattr(val, "audio", None) or val.get("audio")
            try:
                text = transcribe(audio_val.getvalue())
                st.session_state.last_transcription = text
                new_key = "chat_widget_" + str(uuid.uuid4())
                st.session_state.chat_widget_key = new_key
                st.session_state[new_key] = text
            except Exception as exc:
                st.error(f"Transcription failed: {exc}")
        elif getattr(val, "text", None) or (hasattr(val, "get") and val.get("text")):
            text_val = getattr(val, "text", None) or val.get("text")
            st.session_state.text_to_process = text_val
    elif isinstance(val, str) and val.strip():
        st.session_state.text_to_process = val


st.chat_input("How can I help?", key=st.session_state.chat_widget_key, accept_audio=True, on_submit=handle_chat_submit)

if st.session_state.get("last_transcription") and os.environ.get("RAILSENSE_DEBUG", "").lower() in {"1", "true", "yes"}:
    st.info(f"Debug: transcribed text was '{st.session_state.last_transcription}'")
    st.session_state.last_transcription = None

if st.session_state.text_to_process:
    text_to_process = st.session_state.text_to_process
    st.session_state.text_to_process = None

    user_ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.messages.append({
        "role": "user",
        "content": text_to_process,
        "timestamp": user_ts,
    })

    with st.chat_message("user"):
        st.write(text_to_process)
        st.caption(user_ts)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = handle_message(
                    text_to_process,
                    st.session_state.dialogue_state,
                    is_staff=st.session_state.is_staff,
                    session_id=st.session_state.session_id,
                )
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
