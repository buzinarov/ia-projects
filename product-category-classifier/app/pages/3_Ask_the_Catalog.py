import sys
import tempfile
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import chat
from src.data import load_label_maps

st.set_page_config(page_title="Ask the Catalog", page_icon=":speech_balloon:", layout="wide")
st.title("Ask the Catalog")
st.caption(
    "A local agent (Ollama, llama3.1:8b) that decides on its own whether to search the catalog "
    "or classify a photo. No external API calls -- everything here runs on this machine."
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "agent_history" not in st.session_state:
    st.session_state.agent_history = []

with st.sidebar:
    st.subheader("Attach a product photo (optional)")
    uploaded = st.file_uploader("Image", type=["png", "jpg", "jpeg"], key="agent_image")
    maps = load_label_maps()
    gender = st.selectbox("Gender segment", maps["gender_classes"])
    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.session_state.agent_history = []
        st.rerun()

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tool_trace"):
            with st.expander("What the agent did"):
                st.json(msg["tool_trace"])

user_input = st.chat_input("Ask about the catalog, or about an attached photo...")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    image_path = None
    if uploaded is not None:
        tmp_path = Path(tempfile.gettempdir()) / f"agent_upload_{uploaded.name}"
        tmp_path.write_bytes(uploaded.getvalue())
        image_path = str(tmp_path)

    with st.spinner("Thinking..."):
        result = chat(
            user_input,
            history=st.session_state.agent_history,
            image_path=image_path,
            gender=gender,
        )

    st.session_state.agent_history = result["messages"]
    st.session_state.chat_history.append(
        {"role": "assistant", "content": result["reply"], "tool_trace": result["tool_trace"]}
    )
    with st.chat_message("assistant"):
        st.markdown(result["reply"])
        if result["tool_trace"]:
            with st.expander("What the agent did"):
                st.json(result["tool_trace"])
