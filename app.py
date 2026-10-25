"""Streamlit Application for ZentroMart VoiceOps 2026 (Voice + RAG + Tools + Evaluation).

Run locally with:
    streamlit run app.py
"""
import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.agent import ZentroMartAgent
from src.asr_normalizer import format_reply_for_ear, normalize_asr_text
from src.config import (
    GNANI_DEFAULT_LANG,
    GNANI_DEFAULT_VOICE,
    GNANI_TTS_MODEL,
    resolve_dataset_dir,
)
from src.evaluator import run_full_evaluation
from src.step1_baseline import Step1PromptOnlyAgent
from src.voice import voice_turn

st.set_page_config(
    page_title="ZentroMart VoiceOps 2026 — AI Support Agent",
    page_icon="🎧",
    layout="wide",
)


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


_load_env_file()

# Initialize session state
if "dataset_dir" not in st.session_state:
    st.session_state.dataset_dir = str(resolve_dataset_dir())
if "agent" not in st.session_state:
    st.session_state.agent = ZentroMartAgent(dataset_dir=Path(st.session_state.dataset_dir))
if "step1_agent" not in st.session_state:
    st.session_state.step1_agent = Step1PromptOnlyAgent()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_telemetry" not in st.session_state:
    st.session_state.last_telemetry = {}

# -----------------------------------------------------------------------------
# Sidebar: API Key Configuration, Voice Settings, and Dataset Path
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ VoiceOps 2026 Setup")
    st.caption("Configure your Gnani & LLM API keys and test settings.")

    gnani_key_input = st.text_input(
        "GNANI_API_KEY (app.gnani.ai/voice)",
        value=os.environ.get("GNANI_API_KEY", ""),
        type="password",
        help="Required for live Gnani Speech-to-Text (STT) and Text-to-Speech (TTS).",
    )
    if gnani_key_input:
        os.environ["GNANI_API_KEY"] = gnani_key_input.strip()

    llm_provider = st.selectbox(
        "LLM Provider (Steps 1–3)",
        options=["auto (Grounded Engine + API if set)", "gemini", "openai", "groq"],
        index=0,
    )
    api_key_llm = st.text_input(
        "Optional LLM API Key (Gemini / OpenAI / Groq)",
        value=(
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("GROQ_API_KEY")
            or ""
        ),
        type="password",
    )
    if api_key_llm:
        if "gemini" in llm_provider:
            os.environ["GEMINI_API_KEY"] = api_key_llm.strip()
        elif "openai" in llm_provider:
            os.environ["OPENAI_API_KEY"] = api_key_llm.strip()
        elif "groq" in llm_provider:
            os.environ["GROQ_API_KEY"] = api_key_llm.strip()

    st.divider()
    st.subheader("🎙️ Gnani Speech Settings (Step 4)")
    stt_lang = st.selectbox(
        "Input Speech Language (language_code)",
        options=["en-IN", "hi-IN", "bn-IN", "gu-IN", "kn-IN", "ml-IN", "mr-IN", "pa-IN", "ta-IN", "te-IN"],
        index=0,
    )
    tts_voice = st.selectbox(
        "Gnani TTS Voice",
        options=["Kaveri", "Trupti", "Pranav", "Shlok"],
        index=0,
    )
    tts_model = st.selectbox(
        "Gnani TTS Model",
        options=["timbre-v2.5", "timbre-v3.0"],
        index=0,
    )
    agent_stage = st.radio(
        "Active Agent Mode",
        options=[
            "Step 3 & 4: Grounded RAG + Tools + Voice",
            "Step 1: Prompt-Only Baseline (No KB)",
        ],
        index=0,
    )

    if st.button("🔄 Reset Conversation", use_container_width=True):
        st.session_state.agent.reset()
        st.session_state.messages = []
        st.session_state.last_telemetry = {}
        st.rerun()

# -----------------------------------------------------------------------------
# Main Header & Tabs
# -----------------------------------------------------------------------------
st.title("🎧 ZentroMart VoiceOps 2026 — Customer Support Agent")
st.caption(
    "LLM & Generative AI Capstone · Consulting & Analytics Club, IIT Guwahati × Gnani.ai · Reference Date: **20 October 2026**"
)

tab_chat, tab_voice, tab_eval, tab_cheat = st.tabs([
    "💬 1. Multi-Turn Chat & Presets",
    "🎙️ 2. Live Voice Call (Gnani STT/TTS)",
    "📊 3. Step 1–4 Benchmark Evaluation",
    "🗂️ 4. Test Orders & Policy Cheat Sheet",
])


def _handle_turn(user_text: str, is_voice: bool = False) -> dict:
    st.session_state.messages.append({"role": "user", "content": user_text})
    if agent_stage.startswith("Step 1"):
        res = st.session_state.step1_agent.answer_question(user_text)
    else:
        res = st.session_state.agent.respond(user_text, is_voice=is_voice)
    st.session_state.messages.append({
        "role": "assistant",
        "content": res["reply"],
        "chunks": res.get("retrieved_chunks", []),
        "tools": res.get("tool_calls", []),
        "decision": res.get("decision", "answer"),
    })
    st.session_state.last_telemetry = {
        "normalized": normalize_asr_text(user_text),
        **res,
    }
    return res


# =============================================================================
# TAB 1: Multi-Turn Chat & Presets
# =============================================================================
with tab_chat:
    st.markdown("##### Quick-Test Preset Prompts")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("👟 Order 1859 9253 (Plus Return)", use_container_width=True):
        _handle_turn("Hello, mujhe order 1859 9253 ke baare mein help chahiye.")
    if c2.button("📺 Order 8978 2837 (>₹10k TV Damage)", use_container_width=True):
        _handle_turn("Hello, mujhe order 8978 2837 ke baare mein help chahiye.")
    if c3.button("🌙 Midnight Delivery ('I don't know')", use_container_width=True):
        _handle_turn("Can I schedule a delivery at midnight?")
    if c4.button("💳 17 Oct Card Refund + Holiday Math", use_container_width=True):
        _handle_turn("My refund to credit card was started on 17 October 2026. By which date should I have it?")

    col_left, col_right = st.columns([1.6, 1.0])

    with col_left:
        chat_box = st.container(height=420)
        with chat_box:
            if not st.session_state.messages:
                st.info(
                    "👋 Ask any ZentroMart policy question or enter an 8-digit Order ID (e.g. `1859 9253` with phone last-4 `4265`, or `8978 2837` with phone last-4 `4533`)."
                )
            for m in st.session_state.messages:
                with st.chat_message(m["role"]):
                    st.write(m["content"])
                    if m["role"] == "assistant" and (m.get("chunks") or m.get("tools")):
                        badges = " · ".join(
                            [f"`{c}`" for c in m.get("chunks", [])]
                            + [f"🛠️ `{t}`" for t in m.get("tools", [])]
                        )
                        st.caption(f"Evidence & Tools: {badges}")

        user_prompt = st.chat_input("Type in English, Hindi, or Hinglish...")
        if user_prompt:
            _handle_turn(user_prompt, is_voice=False)
            st.rerun()

    with col_right:
        st.subheader("🧠 Agent Brain Inspector")
        ag = st.session_state.agent
        st.markdown(
            f"**Active Order ID:** `{ag.order_id or 'None'}`  \n"
            f"**Caller Verified:** `{'Yes ✅' if ag.verified_order else f'No (Attempts: {ag.verification_attempts}/2)'}`"
        )
        tel = st.session_state.last_telemetry
        if tel:
            st.markdown(f"**Decision:** `{tel.get('decision')}` | **Language:** `{tel.get('language')}`")
            st.markdown(f"**ASR Normalized Input:** `{tel.get('normalized')}`")
            st.markdown("**Retrieved Chunks (`gold_evidence`):**")
            st.code(json.dumps(tel.get("retrieved_chunks", []), indent=2))
            st.markdown("**Executed Tool Calls (`tools.py`):**")
            st.code(json.dumps(tel.get("tool_calls", []), indent=2))
        else:
            st.caption("Send a message to inspect retrieved KB chunks and tool calls.")


# =============================================================================
# TAB 2: Live Voice Call (Gnani STT + Agent + Gnani TTS)
# =============================================================================
with tab_voice:
    st.subheader("🎙️ Step 4: Spoken Customer Turn (Gnani STT → ZentroMart Agent → Gnani TTS)")
    st.markdown(
        "Record your voice directly in the browser or upload a `.wav` file. "
        "Make sure `GNANI_API_KEY` is entered in the sidebar (or in `.env`)."
    )

    v_col1, v_col2 = st.columns(2)
    with v_col1:
        recorded_audio = None
        if hasattr(st, "audio_input"):
            recorded_audio = st.audio_input("Record your voice question (up to 60 seconds)")
        uploaded_audio = st.file_uploader("Or upload a customer audio file (.wav / .mp3)", type=["wav", "mp3", "flac", "ogg"])

        audio_source = recorded_audio or uploaded_audio
        if audio_source is not None and st.button("🚀 Run Gnani Voice Turn", type="primary"):
            if not os.environ.get("GNANI_API_KEY"):
                st.error("Please enter your `GNANI_API_KEY` in the left sidebar or `.env` file first!")
            else:
                with tempfile.TemporaryDirectory() as tmpdir:
                    in_wav = Path(tmpdir) / "customer_input.wav"
                    out_wav = Path(tmpdir) / "reply.wav"
                    in_wav.write_bytes(audio_source.read())
                    try:
                        heard, reply_text, meta = voice_turn(
                            audio_path=str(in_wav),
                            agent=st.session_state.agent,
                            language_code=stt_lang,
                            voice=tts_voice,
                            model=tts_model,
                            output_file=str(out_wav),
                        )
                        st.success(f"**Gnani STT Transcript:** {heard}")
                        st.info(f"**Agent Spoken Reply ({tts_voice}, {tts_model}):** {reply_text}")
                        if out_wav.is_file():
                            st.audio(out_wav.read_bytes(), format="audio/wav")
                        st.json(meta)
                    except Exception as exc:
                        st.error(f"Gnani API call failed: {exc}")

    with v_col2:
        st.markdown("##### Test Step 4 ASR Normalization Without Microphone")
        asr_input = st.selectbox(
            "Select a noisy ASR transcript from practice/questions.csv:",
            options=[
                "so sku ele-zero zero one to pe zentromart assured hai kya claim kaise karein",
                "uh as a zentromart plus member will i be charged the pickup fee for returning shoes i just dont like",
                "uh the kestrel book 14 laptop sku ele-zero zero two two i received yesterday has a crack i want it sorted",
                "my re fund to net banking was started on 14 october two zero two six by which date should i have it",
            ],
        )
        if st.button("Test ASR Transcript"):
            res = st.session_state.agent.answer_question(asr_input, is_voice=True)
            st.markdown(f"**Normalized Query:** `{normalize_asr_text(asr_input)}`")
            st.markdown(f"**Ear-Formatted Reply:** {format_reply_for_ear(res['reply'])}")
            st.markdown(f"**Retrieved Chunks:** `{res['retrieved_chunks']}`")


# =============================================================================
# TAB 3: Step 1–4 Benchmark Evaluation
# =============================================================================
with tab_eval:
    st.subheader("📊 Automated Evaluation on `practice/questions.csv` & `practice/conversations.csv`")
    if st.button("▶️ Run Full Benchmark Evaluation (52 Questions + 23 Conversations)"):
        with st.spinner("Running Step 1, Step 2, Step 3, and Step 4 evaluations..."):
            eval_res = run_full_evaluation(dataset_dir=Path(st.session_state.dataset_dir))
            summary = eval_res["summary"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Step 1 (Prompt-Only)", f"{summary['step1_accuracy_pct']}%", f"{summary['step1_prompt_only_correct']}/52")
            m2.metric("Step 2 (RAG Grounded)", f"{summary['step2_accuracy_pct']}%", f"{summary['step2_rag_correct']}/52")
            m3.metric("Step 3 (Conversations)", f"{summary['step3_turn_accuracy_pct']}%", f"{summary['step3_turns_passed']}/96 turns")
            m4.metric("Step 4 (ASR Voice)", f"{summary['step4_asr_accuracy_pct']}%", f"{summary['step4_asr_correct']}/52")
            st.dataframe(pd.DataFrame(eval_res["step2"]), use_container_width=True)


# =============================================================================
# TAB 4: Test Orders & Policy Cheat Sheet
# =============================================================================
with tab_cheat:
    st.subheader("🗂️ Sample Verifiable Orders for Demo Recording")
    st.markdown(
        "Use any of these order IDs and their matching **4-digit phone suffix** or **6-digit delivery pincode** when recording your 2-minute demo:"
    )
    cheat_df = pd.DataFrame([
        {"Order ID": "1859 9253", "Phone Last 4": "4265", "Pincode": "793001", "Scenario": "Plus Member Sneaker Return (until 27 Oct 2026) + Instant Wallet Refund Follow-Up"},
        {"Order ID": "8978 2837", "Phone Last 4": "4533", "Pincode": "781006", "Scenario": "Damaged 43-inch 4K Smart TV (₹31,649 > ₹10,000) -> Priority Resolution Ticket Handover"},
        {"Order ID": "7936 3550", "Phone Last 4": "2949", "Pincode": "380015", "Scenario": "Festive Sale Jeans Return (30-day window until 13 Nov 2026) + ₹49 Pickup Fee"},
        {"Order ID": "1664 0894", "Phone Last 4": "2108", "Pincode": "395007", "Scenario": "Overdue UPI Refund (RRN 315633825001) -> Escalate via create_ticket()"},
        {"Order ID": "3506 1733", "Phone Last 4": "2743", "Pincode": "411014", "Scenario": "Unshipped Order ('placed') -> Guide to cancel free in My Orders (CAN-1)"},
        {"Order ID": "1591 1102", "Phone Last 4": "9328", "Pincode": "440010", "Scenario": "Shipped Order -> Refuse cancellation (CAN-2), advise refusing parcel at door"},
        {"Order ID": "7746 1439", "Phone Last 4": "1263 (Try 1266 to fail)", "Pincode": "600042", "Scenario": "Failed Verification Test (1 retry allowed, 2nd wrong attempt locks order)"},
    ])
    st.table(cheat_df)
