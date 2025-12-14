"""
MemMachine Simple Chatbot - Streamlit Application

A complete Streamlit-based chatbot that demonstrates MemMachine's persistent
memory capabilities with support for multiple LLM providers.
"""

import json
import time
from pathlib import Path
from typing import cast

import streamlit as st
from gateway_client import delete_profile, ingest_and_rewrite, ingest_memories
from llm import chat, set_model
from model_config import MODEL_CHOICES, MODEL_DISPLAY_NAMES

# Constants
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_PERSONA = "Charlie"
DEFAULT_SESSION_BASE = "Session"
TYPING_SPEED = 0.02
PREVIEW_TEXT_LIMIT = 2000


# ──────────────────────────────────────────────────────────────
# Session Management Functions
# ──────────────────────────────────────────────────────────────


def _generate_session_name(base: str = DEFAULT_SESSION_BASE) -> str:
    """Generate a unique session name."""
    existing = set(st.session_state.get("session_order", []))
    idx = 1
    while True:
        candidate = f"{base} {idx}"
        if candidate not in existing:
            return candidate
        idx += 1


def ensure_session_state() -> None:
    """Initialize session state with default values."""
    if "sessions" not in st.session_state:
        st.session_state.sessions = {}
    if "session_order" not in st.session_state:
        st.session_state.session_order = []
    if (
        "active_session_id" not in st.session_state
        or st.session_state.active_session_id not in st.session_state.sessions
    ):
        default_name = _generate_session_name()
        st.session_state.sessions.setdefault(default_name, {"history": []})
        if default_name not in st.session_state.session_order:
            st.session_state.session_order.append(default_name)
        st.session_state.active_session_id = default_name
    if "session_select" not in st.session_state:
        st.session_state.session_select = st.session_state.active_session_id
    if st.session_state.session_select not in st.session_state.sessions:
        st.session_state.session_select = st.session_state.active_session_id
    st.session_state.setdefault(
        "rename_session_name", st.session_state.active_session_id
    )
    st.session_state.setdefault(
        "rename_session_synced_to", st.session_state.active_session_id
    )
    st.session_state.history = cast(
        list[dict],
        st.session_state.sessions[st.session_state.active_session_id].setdefault(
            "history", []
        ),
    )


def create_session(session_name: str | None = None) -> tuple[bool, str]:
    """Create a new session."""
    ensure_session_state()
    candidate = (session_name or "").strip()
    if not candidate:
        candidate = _generate_session_name()
    if candidate in st.session_state.sessions:
        return False, candidate
    st.session_state.sessions[candidate] = {"history": []}
    st.session_state.session_order.append(candidate)
    st.session_state.active_session_id = candidate
    st.session_state.session_select = candidate
    st.session_state.history = cast(
        list[dict], st.session_state.sessions[candidate]["history"]
    )
    st.session_state.rename_session_name = candidate
    st.session_state.rename_session_synced_to = candidate
    return True, candidate


def rename_session(current_name: str, new_name: str) -> bool:
    """Rename an existing session."""
    ensure_session_state()
    target = new_name.strip()
    if not target or target == current_name:
        return False
    if target in st.session_state.sessions:
        return False
    st.session_state.sessions[target] = st.session_state.sessions.pop(current_name)
    order = st.session_state.session_order
    order[order.index(current_name)] = target
    if st.session_state.active_session_id == current_name:
        st.session_state.active_session_id = target
        st.session_state.session_select = target
    st.session_state.history = cast(
        list[dict],
        st.session_state.sessions[st.session_state.active_session_id]["history"],
    )
    st.session_state.rename_session_name = target
    st.session_state.rename_session_synced_to = target
    return True


def delete_session(session_name: str) -> bool:
    """Delete a session."""
    ensure_session_state()
    if session_name not in st.session_state.sessions:
        return False
    if len(st.session_state.session_order) <= 1:
        return False
    st.session_state.sessions.pop(session_name, None)
    st.session_state.session_order.remove(session_name)
    if st.session_state.active_session_id == session_name:
        st.session_state.active_session_id = st.session_state.session_order[-1]
        st.session_state.session_select = st.session_state.active_session_id
        st.session_state.rename_session_name = st.session_state.active_session_id
        st.session_state.rename_session_synced_to = st.session_state.active_session_id
    st.session_state.history = cast(
        list[dict],
        st.session_state.sessions[st.session_state.active_session_id]["history"],
    )
    return True


# ──────────────────────────────────────────────────────────────
# Message Processing Functions
# ──────────────────────────────────────────────────────────────


def rewrite_message(
    msg: str, persona_name: str, show_rationale: bool, use_memory: bool = True
) -> str:
    """Rewrite message with memory context if enabled."""
    if not use_memory or persona_name.lower() == "control":
        rewritten_msg = msg
        if show_rationale:
            rewritten_msg += (
                " At the beginning of your response, please say the following in ITALIC: "
                "'Persona Rationale: No personalization applied.'. "
                "Begin your answer on the next line."
            )
        return rewritten_msg

    try:
        rewritten_msg = ingest_and_rewrite(user_id=persona_name, query=msg)
        if show_rationale:
            rewritten_msg += (
                " At the beginning of your response, please say the following in ITALIC: "
                "'Persona Rationale: ' followed by 1 sentence about how your reasoning "
                "for how the persona traits influenced this response, also in italics. "
                "Begin your answer on the next line."
            )
    except Exception as e:
        st.error(f"Failed to ingest_and_append message: {e}")
        raise
    print(rewritten_msg)
    return rewritten_msg


def clean_history(history: list[dict], persona: str) -> list[dict]:
    """Clean history to enforce alternating roles and filter by persona."""
    out = []
    for turn in history:
        if turn.get("role") == "user":
            out.append({"role": "user", "content": turn["content"]})
        elif turn.get("role") == "assistant" and turn.get("persona") == persona:
            out.append({"role": "assistant", "content": turn["content"]})
    cleaned = []
    last_role = None
    for msg in out:
        if msg["role"] != last_role:
            cleaned.append(msg)
            last_role = msg["role"]
    return cleaned


def append_user_turn(msgs: list[dict], new_user_msg: str) -> list[dict]:
    """Append or replace the last user message."""
    if msgs and msgs[-1]["role"] == "user":
        msgs[-1] = {"role": "user", "content": new_user_msg}
    else:
        msgs.append({"role": "user", "content": new_user_msg})
    return msgs


def typewriter_effect(text: str, speed: float = TYPING_SPEED):
    """Generator that yields text word by word to create a typing effect."""
    words = text.split(" ")
    for i, word in enumerate(words):
        if i == 0:
            yield word
        else:
            yield " " + word
        time.sleep(speed)


# ──────────────────────────────────────────────────────────────
# UI Component Functions
# ──────────────────────────────────────────────────────────────


def load_css() -> None:
    """Load CSS from styles.css file."""
    css_path = Path(__file__).parent / "styles.css"
    try:
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


def render_header() -> None:
    """Render the MemMachine header with links."""
    st.markdown(
        """
    <div class="memmachine-header-wrapper">
      <div class="memmachine-header-links">
        <span class="powered-by">Powered by MemMachine</span>
        <a href="https://memmachine.ai/" target="_blank" title="MemMachine">
          <img src="https://avatars.githubusercontent.com/u/226739620?s=48&v=4" alt="MemMachine logo"/>
        </a>
        <a href="https://github.com/MemMachine/MemMachine" target="_blank" title="GitHub Repository">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
          </svg>
        </a>
        <a href="https://discord.gg/usydANvKqD" target="_blank" title="Discord Community">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
            <path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028c.462-.63.874-1.295 1.226-1.994a.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.128 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.946 2.418-2.157 2.418z"/>
          </svg>
        </a>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_memory_status(memmachine_enabled: bool) -> None:
    """Render the memory status indicator."""
    status_emoji = "🧠" if memmachine_enabled else "⚪"
    status_text = "MemMachine Active" if memmachine_enabled else "No Memory Mode"
    st.markdown(
        f"""
        <div class="memory-status-indicator">
            <span class="memory-status-text">
                {status_emoji} <strong>{status_text}</strong>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_comparison_banner() -> None:
    """Render the comparison feature banner."""
    st.markdown(
        """
    <div class="comparison-banner">
        <div class="comparison-banner-content">
            <span style="font-size: 1.5rem;">⚖️</span>
            <div>
                <div class="comparison-banner-title">Side-by-Side Comparison with Control Persona</div>
                <div class="comparison-banner-subtitle">Compare MemMachine responses vs Control Persona (no memory)</div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_comparison_header() -> None:
    """Render the comparison header."""
    st.markdown(
        """
    <div class="comparison-header">
        <span class="comparison-header-icon">⚖️</span>
        <span>Side-by-Side Comparison</span>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_comparison_panel(
    label: str, response: str, is_memmachine: bool, is_new: bool
) -> None:
    """Render a comparison panel (MemMachine or Control)."""
    panel_class = (
        "comparison-panel-memmachine" if is_memmachine else "comparison-panel-control"
    )
    icon = "🧠" if is_memmachine else "⚪"
    title_class = "comparison-panel-title"

    st.markdown(
        f"""
    <div class="{panel_class}">
        <div class="comparison-panel-header">
            <span class="comparison-panel-icon">{icon}</span>
            <strong class="{title_class}">{label}</strong>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if is_new:
        st.write_stream(typewriter_effect(response))
    else:
        st.markdown(f'<div class="answer">{response}</div>', unsafe_allow_html=True)


def render_session_list(active_session: str) -> None:
    """Render the list of sessions in the sidebar."""
    session_options = st.session_state.session_order

    for session_name in session_options:
        is_active = session_name == active_session
        button_col, menu_col = st.columns([0.8, 0.2])

        with button_col:
            if (
                st.button(
                    session_name,
                    key=f"session_button_{session_name}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                )
                and not is_active
            ):
                st.session_state.active_session_id = session_name
                st.session_state.session_select = session_name
                st.session_state.history = cast(
                    list[dict],
                    st.session_state.sessions[session_name]["history"],
                )
                st.session_state.rename_session_name = session_name
                st.session_state.rename_session_synced_to = session_name
                st.rerun()

        with menu_col:
            if hasattr(st, "popover"):
                menu_container = st.popover("⋯", use_container_width=True)
            else:
                menu_container = st.expander(
                    "⋯", expanded=False, key=f"session_actions_{session_name}"
                )
            with menu_container:
                render_session_menu(session_name)


def render_session_menu(session_name: str) -> None:
    """Render the menu for session actions (rename/delete)."""
    st.markdown(f"**Actions for {session_name}**")
    rename_value = st.text_input(
        "Rename session",
        value=session_name,
        key=f"rename_session_input_{session_name}",
    )

    if st.button(
        "Rename",
        use_container_width=True,
        key=f"rename_session_button_{session_name}",
    ):
        rename_target = rename_value.strip()
        if not rename_target:
            st.warning("Enter a session name to rename.")
        elif rename_target == session_name:
            st.info("Session name unchanged.")
        elif rename_target in st.session_state.sessions:
            st.warning(f"Session '{rename_target}' already exists.")
        elif rename_session(session_name, rename_target):
            st.success(f"Session renamed to '{rename_target}'.")
            st.rerun()
        else:
            st.error("Unable to rename session. Please try again.")

    st.divider()
    if st.button(
        "Delete session",
        use_container_width=True,
        type="secondary",
        key=f"delete_session_button_{session_name}",
    ):
        if delete_session(session_name):
            new_active = st.session_state.active_session_id
            st.session_state.session_select = new_active
            st.session_state.rename_session_name = new_active
            st.session_state.rename_session_synced_to = new_active
            st.success(f"Session '{session_name}' deleted.")
            st.rerun()
        else:
            st.warning("Cannot delete the last remaining session.")


# ──────────────────────────────────────────────────────────────
# Main Application
# ──────────────────────────────────────────────────────────────


def render_sidebar() -> tuple[str, bool, bool, bool]:
    """Render the sidebar and return configuration values."""
    st.markdown("#### Sessions")

    active_session = st.session_state.active_session_id
    if st.session_state.rename_session_synced_to != active_session:
        st.session_state.rename_session_name = active_session
        st.session_state.rename_session_synced_to = active_session

    render_session_list(active_session)

    with st.form("create_session_form", clear_on_submit=True):
        new_session_name = st.text_input(
            "New session name",
            key="create_session_name",
            placeholder="Leave blank for automatic name",
        )
        if st.form_submit_button("Create session", use_container_width=True):
            success, created_name = create_session(new_session_name)
            if success:
                st.success(f"Session '{created_name}' created.")
                st.rerun()
            else:
                st.warning(f"Session '{created_name}' already exists.")

    st.divider()

    # Model selection
    st.markdown("#### Choose Model")
    display_options = [MODEL_DISPLAY_NAMES[model] for model in MODEL_CHOICES]
    selected_display = st.selectbox(
        "Choose Model", display_options, index=0, label_visibility="collapsed"
    )
    model_id = next(
        model
        for model, display in MODEL_DISPLAY_NAMES.items()
        if display == selected_display
    )
    set_model(model_id)

    # Persona selection
    st.markdown("#### User Identity")
    selected_persona = st.selectbox(
        "Choose user persona",
        ["Charlie", "Jing", "Charles", "Control"],
        label_visibility="collapsed",
    )
    custom_persona = st.text_input("Or enter your name", "")
    persona_name = (
        custom_persona.strip() if custom_persona.strip() else selected_persona
    )

    # Memory toggle
    if "memmachine_enabled" not in st.session_state:
        st.session_state.memmachine_enabled = True
    if "compare_personas" not in st.session_state:
        st.session_state.compare_personas = True

    memmachine_enabled = st.checkbox(
        "Enable MemMachine",
        value=st.session_state.memmachine_enabled,
        help=(
            "Enable MemMachine's persistent memory system. "
            "When unchecked, the AI will respond without memory (Control Persona mode)."
        ),
    )
    st.session_state.memmachine_enabled = memmachine_enabled

    if memmachine_enabled:
        render_comparison_banner()
        compare_personas = st.checkbox(
            "🔄 Compare with control persona",
            value=st.session_state.compare_personas,
            help=(
                "Enable side-by-side comparison to see how MemMachine's persistent "
                "memory enhances responses compared to the control persona (no memory)"
            ),
        )
        st.session_state.compare_personas = compare_personas
    else:
        compare_personas = False

    show_rationale = st.checkbox("Show Persona Rationale")

    st.divider()

    # Action buttons
    if st.button("Clear chat", use_container_width=True):
        active = st.session_state.active_session_id
        st.session_state.sessions[active]["history"].clear()
        st.session_state.history = cast(
            list[dict],
            st.session_state.sessions[active]["history"],
        )
        st.rerun()

    if st.button("Delete Profile", use_container_width=True):
        #########################################################
        # WAIT FOR CEDRIC TO CHANGE THE ID PARAM OF DELETE ENDPOINT FROM STRING TO LIST
        #########################################################
        success = delete_profile(persona_name)
        active = st.session_state.active_session_id
        st.session_state.sessions[active]["history"].clear()
        st.session_state.history = cast(
            list[dict],
            st.session_state.sessions[active]["history"],
        )
        if success:
            st.success(f"Profile for '{persona_name}' deleted.")
        else:
            st.error(f"Failed to delete profile for '{persona_name}'.")

    st.divider()

    return persona_name, memmachine_enabled, compare_personas, show_rationale


def render_memory_import_section(persona_name: str) -> None:
    """Render the memory import section."""
    if "memories_preview" not in st.session_state:
        st.session_state.memories_preview = None
    if "imported_memories_text" not in st.session_state:
        st.session_state.imported_memories_text = ""

    with st.expander(
        "📋 Load Previous Memories (Import from ChatGPT, etc.)", expanded=False
    ):
        st.markdown(
            "**Paste your conversation history or memories from external sources "
            "(e.g., ChatGPT, other AI chats)**"
        )

        imported_text = st.text_area(
            "Paste your memories/conversations here",
            value=st.session_state.imported_memories_text,
            height=200,
            placeholder=(
                "Example:\nUser: What is machine learning?\n"
                "Assistant: Machine learning is...\n\n"
                "User: Can you explain neural networks?\n"
                "Assistant: Neural networks are..."
            ),
            help=(
                "Paste any conversation history, notes, or context you want the AI to remember. "
                "These will be ingested into MemMachine's memory system and available for "
                "future conversations."
            ),
            key="import_memories_textarea",
        )

        uploaded_file = st.file_uploader(
            "Or upload a text file",
            type=["txt", "md", "json"],
            help="Upload a text file containing your conversation history or memories",
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.type == "application/json":
                    file_content = json.loads(uploaded_file.read().decode("utf-8"))
                    imported_text = str(file_content)
                else:
                    imported_text = uploaded_file.read().decode("utf-8")
                st.session_state.imported_memories_text = imported_text
                st.success("File loaded successfully!")
            except Exception as e:
                st.error(f"Error reading file: {e}")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("👁️ Preview", use_container_width=True, key="preview_memories"):
                if imported_text and imported_text.strip():
                    st.session_state.memories_preview = imported_text
                    st.session_state.imported_memories_text = imported_text
                    st.rerun()
                else:
                    st.warning("Please paste or upload some memories first.")

        with col2:
            if st.button(
                "💉 Ingest into MemMachine",
                use_container_width=True,
                key="inject_memories_direct",
            ):
                if imported_text and imported_text.strip():
                    if persona_name and persona_name != "Control":
                        with st.spinner("Ingesting memories into MemMachine..."):
                            success = ingest_memories(persona_name, imported_text)
                            if success:
                                st.session_state.imported_memories_text = imported_text
                                st.success(
                                    "✅ Memories successfully ingested into MemMachine! "
                                    "They are now part of your memory system."
                                )
                            else:
                                st.error(
                                    "❌ Failed to ingest memories. Please try again."
                                )
                    else:
                        st.warning("Please select a persona to ingest memories.")
                    st.rerun()
                else:
                    st.warning("Please paste or upload some memories first.")

    # Show preview if memories are loaded
    if st.session_state.memories_preview:
        with st.expander("📋 Preview Imported Memories", expanded=True):
            memories = st.session_state.memories_preview
            preview_text = str(memories)[:PREVIEW_TEXT_LIMIT]

            if preview_text:
                st.text_area(
                    "Memories Preview",
                    preview_text,
                    height=200,
                    disabled=True,
                    key="memories_preview_text",
                )
                st.caption(f"Total length: {len(str(memories))} characters")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button(
                        "💉 Ingest into MemMachine",
                        use_container_width=True,
                        key="inject_memories_from_preview",
                    ):
                        if persona_name and persona_name != "Control":
                            with st.spinner("Ingesting memories into MemMachine..."):
                                success = ingest_memories(
                                    persona_name, str(st.session_state.memories_preview)
                                )
                            if success:
                                st.success(
                                    "✅ Memories successfully ingested into MemMachine! "
                                    "They are now part of your memory system."
                                )
                            else:
                                st.error(
                                    "❌ Failed to ingest memories. Please try again."
                                )
                            st.rerun()
                        else:
                            st.warning("Please select a persona to ingest memories.")
                            st.rerun()
                with col2:
                    if st.button(
                        "🗑️ Clear",
                        use_container_width=True,
                        key="clear_memories_preview",
                    ):
                        st.session_state.memories_preview = None
                        st.session_state.imported_memories_text = ""
                        st.rerun()
            else:
                st.info("No memories to preview.")
                st.session_state.memories_preview = None


def render_chat_history() -> None:
    """Render the chat history."""
    for turn in st.session_state.history:
        if turn.get("role") == "user":
            st.chat_message("user").write(turn["content"])
        elif turn.get("role") == "assistant":
            with st.chat_message("assistant"):
                if turn.get("is_new", False):
                    st.write_stream(typewriter_effect(turn["content"]))
                    turn["is_new"] = False
                else:
                    st.write(turn["content"])
        elif turn.get("role") == "assistant_all":
            content_items = list(turn["content"].items())
            is_new = turn.get("is_new", False)
            if len(content_items) >= 2:
                render_comparison_header()
                cols = st.columns([1, 0.03, 1])
                persona_label, persona_response = content_items[0]
                control_label, control_response = content_items[1]
                with cols[0]:
                    render_comparison_panel(
                        persona_label,
                        persona_response,
                        is_memmachine=True,
                        is_new=is_new,
                    )
                with cols[1]:
                    st.markdown(
                        '<div class="vertical-divider"></div>', unsafe_allow_html=True
                    )
                with cols[2]:
                    render_comparison_panel(
                        control_label,
                        control_response,
                        is_memmachine=False,
                        is_new=is_new,
                    )
            else:
                for label, response in content_items:
                    st.markdown(f"**{label}**")
                    if is_new:
                        st.write_stream(typewriter_effect(response))
                    else:
                        st.markdown(
                            f'<div class="answer">{response}</div>',
                            unsafe_allow_html=True,
                        )
            if is_new:
                turn["is_new"] = False


def process_user_message(
    msg: str,
    persona_name: str,
    memmachine_enabled: bool,
    compare_personas: bool,
    show_rationale: bool,
) -> None:
    """Process a user message and generate response(s)."""
    st.session_state.history.append({"role": "user", "content": msg})

    if compare_personas and memmachine_enabled:
        all_answers = {}

        # MemMachine response
        rewritten_msg = rewrite_message(
            msg, persona_name, show_rationale, use_memory=True
        )
        msgs = clean_history(st.session_state.history, persona_name)
        msgs = append_user_turn(msgs, rewritten_msg)
        try:
            txt, _, _, _ = chat(msgs, persona_name)
            all_answers[persona_name] = txt
        except ValueError as e:
            st.error(f"❌ {e!s}")
            st.stop()

        print("##########REWRITTEN MSG##########", rewritten_msg)

        # Control response
        rewritten_msg_control = rewrite_message(
            msg, "Control", show_rationale, use_memory=False
        )
        msgs_control = clean_history(st.session_state.history, "Control")
        msgs_control = append_user_turn(msgs_control, rewritten_msg_control)
        try:
            txt_control, _, _, _ = chat(msgs_control, "Arnold")
            all_answers["Control"] = txt_control
        except ValueError as e:
            st.error(f"❌ {e!s}")
            st.stop()

        st.session_state.history.append(
            {
                "role": "assistant_all",
                "axis": "role",
                "content": all_answers,
                "is_new": True,
            }
        )
    else:
        # Single response
        rewritten_msg = rewrite_message(
            msg, persona_name, show_rationale, use_memory=memmachine_enabled
        )
        msgs = clean_history(st.session_state.history, persona_name)
        msgs = append_user_turn(msgs, rewritten_msg)
        try:
            txt, _, _, _ = chat(
                msgs,
                "Arnold"
                if persona_name == "Control" or not memmachine_enabled
                else persona_name,
            )
            st.session_state.history.append(
                {
                    "role": "assistant",
                    "persona": persona_name,
                    "content": txt,
                    "is_new": True,
                }
            )
        except ValueError as e:
            st.error(f"❌ {e!s}")
            st.stop()

    st.rerun()


def main() -> None:
    """Main application entry point."""
    st.set_page_config(page_title="MemMachine Chatbot", layout="wide")

    load_css()
    ensure_session_state()
    render_header()

    # Sidebar configuration
    with st.sidebar:
        persona_name, memmachine_enabled, compare_personas, show_rationale = (
            render_sidebar()
        )

    # Memory import section
    render_memory_import_section(persona_name)

    # Chat input
    msg = st.chat_input("Type your message…")
    if msg:
        process_user_message(
            msg, persona_name, memmachine_enabled, compare_personas, show_rationale
        )

    # Memory status and chat history
    render_memory_status(memmachine_enabled)
    render_chat_history()


if __name__ == "__main__":
    main()
