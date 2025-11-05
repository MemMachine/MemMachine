import os
import streamlit as st
from agents.advisor_buddy import make_advisor_buddy
from utils.security import sanitize_user_id

# ---------- helpers ----------
def extract_final_text(result) -> str:
    # Prefer the last assistant message if the SDK exposes messages
    msgs = getattr(result, "messages", None)
    if isinstance(msgs, list) and msgs:
        for m in reversed(msgs):
            role = getattr(m, "role", None)
            content = getattr(m, "content", None)
            if role == "assistant" and content:
                return content
    # Fallbacks for different SDK versions
    for attr in ("output_text", "outputs_text", "text"):
        val = getattr(result, attr, None)
        if isinstance(val, str) and val.strip():
            return val
    # Last resort
    return str(result)

def ask_buddy(prompt: str) -> str:
    try:
        res = st.session_state.buddy(prompt)
        return extract_final_text(res).strip()
    except Exception as e:
        return f"⚠️ Error: {e}"

# ---------- boot ----------
st.set_page_config(page_title="☕ Morning Brief + MemMachine", layout="wide")

# Generate unique user ID for this session
# Option 1: Use query parameter (e.g., http://localhost:8501?user=anirudh)
# Option 2: Let user enter their name in sidebar
if "user_id" not in st.session_state:
    query_params = st.query_params
    if "user" in query_params:
        # Use URL parameter as user ID - sanitize for security
        raw_user_id = str(query_params["user"])
        st.session_state.user_id = sanitize_user_id(raw_user_id)
        st.session_state.user_id_source = "url"
    else:
        # Generate a random ID for now (will be replaced when user enters name)
        import uuid
        st.session_state.user_id = str(uuid.uuid4())[:8]
        st.session_state.user_id_source = "temp"

if "buddy" not in st.session_state:
    st.session_state.buddy = make_advisor_buddy(user_id=st.session_state.user_id)
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of {"role":"user|assistant","content": str}

# ---------- sidebar ----------
with st.sidebar:
    st.subheader("🧠 Memory System")
    
    # User ID management
    if st.session_state.get("user_id_source") == "temp":
        st.warning("⚠️ Using temporary ID - memories won't persist!")
        st.caption("Enter your name below to enable persistent memory:")
        
        user_name_input = st.text_input("Your Name", key="name_input", placeholder="e.g., anirudh")
        if st.button("Set Persistent ID", use_container_width=True):
            if user_name_input:
                # Sanitize user input for security
                new_user_id = sanitize_user_id(user_name_input)
                st.session_state.user_id = new_user_id
                st.session_state.user_id_source = "manual"
                # Recreate agent with new user ID
                st.session_state.buddy = make_advisor_buddy(user_id=new_user_id)
                st.success(f"✅ Set persistent ID: {new_user_id}")
                st.rerun()
    else:
        st.success(f"✅ Persistent User: **{st.session_state.user_id}**")
        st.caption("Your memories will persist across sessions!")
        if st.button("Change User", use_container_width=True):
            st.session_state.user_id_source = "temp"
            st.rerun()
    
    st.markdown("---")
    
    # MemMachine status
    if st.session_state.buddy.memmachine_enabled:
        st.success("✅ MemMachine Connected")
        if st.session_state.buddy.user_name:
            st.info(f"📝 Remembered Name: {st.session_state.buddy.user_name}")
    else:
        st.warning("⚠️ MemMachine Offline")
        st.caption("Using session-only memory")
    
    st.markdown("---")
    st.subheader("Settings")
    tav = os.getenv("TAVILY_API_KEY")
    st.write("🔑 Tavily:", "✅" if tav else "❌")
    st.caption("Uses Tavily + MemMachine")
    
    st.markdown("---")
    st.markdown("**Quick actions**")
    if st.button("☀️ Daily Brief", use_container_width=True):
        st.session_state.chat.append({"role": "user", "content": "Give me my personalized morning news brief"})
        reply = ask_buddy("Give me my personalized morning news brief")
        st.session_state.chat.append({"role": "assistant", "content": reply})
    
    st.markdown("---")
    st.markdown("### 📋 Features")
    st.markdown("""
    - 🗣️ Natural conversation
    - 📰 Real-time news (Tavily)
    - 💾 Persistent memory
    - 🎯 Context tracking
    - 😊 Personalization
    """)

# ---------- header ----------
st.title("☕ Morning Brief + 🧠 MemMachine")
st.caption("Multi-Agent System: AdvisorBuddy 🎙️ · MemoryKeeper 🧠 · NewsScout 📰")
st.markdown("---")

# Dynamic category buttons removed - use natural language input instead
# Categories are discovered from user preferences stored in memory, not hardcoded

st.markdown("")

# ---------- chat history ----------
for msg in st.session_state.chat:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])

# ---------- input ----------
prompt = st.chat_input("Ask for news or any topic (e.g., 'AI breakthroughs', 'cricket', 'markets')")
if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        with st.spinner("Fetching live updates…"):
            reply = ask_buddy(prompt)
        st.session_state.chat.append({"role": "assistant", "content": reply})
        st.markdown(reply)
