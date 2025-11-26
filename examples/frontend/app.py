from typing import cast

import streamlit as st
from gateway_client import delete_profile, ingest_and_rewrite
from llm import chat, set_model
from model_config import MODEL_CHOICES, MODEL_TO_PROVIDER


def rewrite_message(
    msg: str, persona_name: str, show_rationale: bool, skip_rewrite: bool
) -> str:
    if skip_rewrite:
        rewritten_msg = msg
        if show_rationale:
            rewritten_msg += " At the beginning of your response, please say the following in ITALIC: 'Persona Rationale: No personalization applied.'. Begin your answer on the next line."
    else:
        try:
            rewritten_msg = ingest_and_rewrite(
                user_id=persona_name, query=msg, model_type=provider
            )
            if show_rationale:
                rewritten_msg += " At the beginning of your response, please say the following in ITALIC: 'Persona Rationale: ' followed by 1 sentence about how your reasoning for how the persona traits influenced this response, also in italics. Begin your answer on the next line."

        except Exception as e:
            st.error(f"Failed to ingest_and_append message: {e}")
            raise
    return rewritten_msg


# ──────────────────────────────────────────────────────────────
# Page setup & CSS
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="MemMachine Chatbot", layout="wide")
with open("./styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("./assets/memmachine_logo.png", use_container_width=True)

    st.markdown("#### Choose Model")

    model_id = st.selectbox(
        "Choose Model", MODEL_CHOICES, index=0, label_visibility="collapsed"
    )
    provider = MODEL_TO_PROVIDER[model_id]
    set_model(model_id)

    st.markdown("#### Choose user persona")
    selected_persona = st.selectbox(
        "Choose user persona",
        ["Charlie", "Jing", "Charles", "Control"],
        label_visibility="collapsed",
    )
    custom_persona = st.text_input("Or enter your name", "")
    persona_name = (
        custom_persona.strip() if custom_persona.strip() else selected_persona
    )

    skip_rewrite = st.checkbox("Skip Rewrite")
    compare_personas = st.checkbox("Compare with Control persona")
    show_rationale = st.checkbox("Show Persona Rationale")

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    if st.button("Delete Profile", use_container_width=True):
        success = delete_profile(persona_name)
        st.session_state.history = []
        if success:
            st.success(f"Profile for '{persona_name}' deleted.")
        else:
            st.error(f"Failed to delete profile for '{persona_name}'.")
    st.divider()

# ──────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = cast(list[dict], [])


# ──────────────────────────────────────────────────────────────
# Enforce alternating roles
# ──────────────────────────────────────────────────────────────
def clean_history(history: list[dict], persona: str) -> list[dict]:
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
    if msgs and msgs[-1]["role"] == "user":
        msgs[-1] = {"role": "user", "content": new_user_msg}
    else:
        msgs.append({"role": "user", "content": new_user_msg})
    return msgs


# ──────────────────────────────────────────────────────────────
# Title
# ──────────────────────────────────────────────────────────────
st.title("MemMachine Chatbot")

# ──────────────────────────────────────────────────────────────
# Chat logic
# ──────────────────────────────────────────────────────────────
msg = st.chat_input("Type your message…")
if msg:
    st.session_state.history.append({"role": "user", "content": msg})
    # rewritten_msg = "Use the persona profile to personalize your naswer only when applicable.\n"
    if compare_personas:
        all_answers = {}
        rewritten_msg = rewrite_message(msg, persona_name, show_rationale, False)
        msgs = clean_history(st.session_state.history, persona_name)
        msgs = append_user_turn(msgs, rewritten_msg)
        txt, lat, tok, tps = chat(msgs, persona_name)
        all_answers[persona_name] = txt

        rewritten_msg_control = rewrite_message(msg, "Control", show_rationale, True)
        msgs_control = clean_history(st.session_state.history, "Control")
        msgs_control = append_user_turn(msgs_control, rewritten_msg_control)
        txt_control, lat, tok, tps = chat(msgs_control, "Arnold")
        all_answers["Control"] = txt_control

        st.session_state.history.append(
            {"role": "assistant_all", "axis": "role", "content": all_answers}
        )
    else:
        rewritten_msg = rewrite_message(msg, persona_name, show_rationale, skip_rewrite)
        msgs = clean_history(st.session_state.history, persona_name)
        msgs = append_user_turn(msgs, rewritten_msg)
        txt, lat, tok, tps = chat(
            msgs, "Arnold" if persona_name == "Control" else persona_name
        )
        st.session_state.history.append(
            {"role": "assistant", "persona": persona_name, "content": txt}
        )

# ──────────────────────────────────────────────────────────────
# Chat history display
# ──────────────────────────────────────────────────────────────
for turn in st.session_state.history:
    if turn.get("role") == "user":
        st.chat_message("user").write(turn["content"])
    elif turn.get("role") == "assistant":
        st.chat_message("assistant").write(turn["content"])
    elif turn.get("role") == "assistant_all":
        content_items = list(turn["content"].items())
        if len(content_items) >= 2:
            cols = st.columns([1, 0.03, 1])
            persona_label, persona_response = content_items[0]
            control_label, control_response = content_items[1]
            with cols[0]:
                st.markdown(f"**{persona_label}**")
                st.markdown(
                    f'<div class="answer">{persona_response}</div>',
                    unsafe_allow_html=True,
                )
            with cols[1]:
                st.markdown(
                    '<div class="vertical-divider"></div>', unsafe_allow_html=True
                )
            with cols[2]:
                st.markdown(f"**{control_label}**")
                st.markdown(
                    f'<div class="answer">{control_response}</div>',
                    unsafe_allow_html=True,
                )
        else:
            for label, response in content_items:
                st.markdown(f"**{label}**")
                st.markdown(
                    f'<div class="answer">{response}</div>', unsafe_allow_html=True
                )
