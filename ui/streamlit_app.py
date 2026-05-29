import base64
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_ICON_PATH = _REPO_ROOT / "toy-train.png"
load_dotenv(_REPO_ROOT / ".env")

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chatbot.dialogue import handle_message, init_dialogue_state, GREETING, STAFF_GREETING
from db.conversations import get_sessions, get_session_messages
from db.staff import authenticate, ensure_default_user
from voice.transcriber import transcribe


@st.cache_data
def _train_icon_b64() -> str:
    return base64.b64encode(_TRAIN_ICON_PATH.read_bytes()).decode()

try:
    ensure_default_user()
except Exception:
    pass

st.set_page_config(page_title="RailSense", page_icon=str(_TRAIN_ICON_PATH), initial_sidebar_state="expanded")

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


# ── UI helpers ────────────────────────────────────────────────────────────────

def _spinner_text(dialogue_state: dict, user_input: str, is_staff: bool = False) -> str:
    """Return a task-specific spinner label so users know what's happening."""
    if is_staff:
        return "📋 Searching contingency plans..."
    active = dialogue_state.get("active_task")
    if active == "ticket_search":
        return "🔍 Searching live fares..."
    if active == "delay_prediction":
        return "⏱️ Running delay prediction..."
    if active == "contingency":
        return "📋 Searching contingency plans..."
    # No active task yet -- peek at keywords in the message
    msg = user_input.lower()
    if any(w in msg for w in ("ticket", "fare", "book", "journey", "travel", "cheapest", "single", "return")):
        return "🔍 Searching live fares..."
    if any(w in msg for w in ("delay", "late", "running", "behind", "predict", "arrival")):
        return "⏱️ Running delay prediction..."
    return "💭 Thinking..."


def _slot_progress(dialogue_state: dict) -> None:
    """Show a subtle progress bar while the bot is collecting multi-turn fields."""
    active = dialogue_state.get("active_task")
    if active == "ticket_search":
        ts = dialogue_state.get("ticket_state", {})
        required = [("origin", "origin"), ("destination", "destination"), ("departure_time", "departure time")]
        filled = [label for key, label in required if ts.get(key)]
        n, total = len(filled), len(required)
        if 0 < n < total:
            st.progress(n / total, text=f"Journey details  collected: {', '.join(filled)}")
    elif active == "delay_prediction":
        ds = dialogue_state.get("delay_state", {})
        required = [
            ("station", "current station"), ("destination", "destination"),
            ("current_delay_minutes", "delay minutes"), ("planned_arrival_time", "arrival time"),
        ]
        filled = [label for key, label in required if ds.get(key)]
        n, total = len(filled), len(required)
        if 0 < n < total:
            st.progress(n / total, text=f"Delay details  collected: {', '.join(filled)}")


def _render_assistant_extras(msg: dict) -> None:
    journeys = msg.get("journeys")
    prediction = msg.get("prediction")
    weather = msg.get("weather")
    debug = msg.get("debug")
    if journeys:
        _render_journeys(journeys)
    if prediction:
        st.json(prediction)
    if weather:
        with st.container(border=True):
            st.markdown(
                f"**{weather.get('icon', '')} {weather.get('location', '')}**  "
                f"{weather.get('condition', '')} {weather.get('temperature_c', '')}°C | "
                f"Wind {weather.get('wind_speed_kmh', '')} km/h"
            )
    if debug:
        with st.expander("Debug details"):
            st.json(debug)


def _render_journeys(journeys: list) -> None:
    """Render journey results as a clean table with clickable booking links."""
    if not journeys:
        return
    df = pd.DataFrame(journeys)
    col_cfg: dict = {}
    name_map = {
        "origin":      "From",
        "destination": "To",
        "departure":   "Departs",
        "arrival":     "Arrives",
        "price":       "Price",
    }
    for key, label in name_map.items():
        if key in df.columns:
            col_cfg[key] = st.column_config.TextColumn(label)
    if "link" in df.columns:
        col_cfg["link"] = st.column_config.LinkColumn("Book", display_text="🎫 Book ticket")
    st.dataframe(df, column_config=col_cfg, hide_index=True, use_container_width=True)


# Sidebar: conversation history
with st.sidebar:
    if st.button("+ New chat", key="sidebar_new_chat", use_container_width=True, type="primary"):
        st.session_state.session_id = str(uuid.uuid4())
        greeting = STAFF_GREETING if st.session_state.is_staff else GREETING
        st.session_state.messages = [{"role": "assistant", "content": greeting}]
        st.session_state.dialogue_state = init_dialogue_state()
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


# Login page
if st.session_state.page == "login":
    if st.button("Back", key="login_back"):
        st.session_state.page = "chat"
        st.rerun()

    st.title("Staff Login")
    st.text_input("Username", key="login_username")
    st.text_input("Password", type="password", key="login_password")
    if st.button("Log in", key="login_submit", type="primary"):
        username = st.session_state.get("login_username", "")
        password = st.session_state.get("login_password", "")
        if authenticate(username, password):
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
    st.markdown(
        f'<h2 style="display:flex;align-items:center;gap:10px;margin:0;">'
        f'<img src="data:image/png;base64,{_train_icon_b64()}" width="36" height="36" style="display:inline-block;">'
        f'RailSense</h2>',
        unsafe_allow_html=True,
    )
with col_auth:
    if st.session_state.is_staff:
        st.markdown(
            '<div style="background:#198754;color:white;padding:4px 10px;border-radius:16px;'
            'font-size:0.78rem;font-weight:600;text-align:center;margin-bottom:4px;"> Staff</div>',
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

if st.session_state.is_staff:
    st.info(
        " **Staff Portal**  You have access to operational contingency guidance and disruption plans.",
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant":
            _render_assistant_extras(msg)
        if msg.get("timestamp"):
            st.caption(msg["timestamp"])


# Quick-start prompt chips  only shown on a fresh chat (just the greeting)
_is_fresh = (
    len(st.session_state.messages) == 1
    and st.session_state.messages[0]["role"] == "assistant"
)
if _is_fresh and not st.session_state.is_staff:
    _chips = [
        ("Find a cheap ticket", "I want to find the cheapest train ticket for my journey"),
        ("My train is delayed", "My train is delayed, can you predict arrival?"),
    ]
    _chip_cols = st.columns(len(_chips))
    for _i, (_label, _prompt) in enumerate(_chips):
        with _chip_cols[_i]:
            if st.button(_label, key=f"chip_{_i}", use_container_width=True):
                st.session_state.text_to_process = _prompt
                st.rerun()


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


_slot_progress(st.session_state.dialogue_state)

_placeholder = (
    "Ask about station contingency plans or disruptions..."
    if st.session_state.is_staff
    else "Ask about tickets, delays, or your journey..."
)
st.chat_input(_placeholder, key=st.session_state.chat_widget_key, accept_audio=True, on_submit=handle_chat_submit)

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
        with st.spinner(_spinner_text(st.session_state.dialogue_state, text_to_process, is_staff=st.session_state.is_staff)):
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
        _render_assistant_extras(response)
        bot_ts = datetime.now().strftime("%H:%M:%S")
        st.caption(bot_ts)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response["message"],
        "journeys": response.get("journeys"),
        "prediction": response.get("prediction"),
        "weather": response.get("weather"),
        "debug": response.get("debug"),
        "timestamp": bot_ts,
    })
    st.rerun()
