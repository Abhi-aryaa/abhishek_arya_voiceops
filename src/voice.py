"""Step 4: Voice Layer using Gnani Official Python SDK (gnani-vachana)."""
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .agent import ZentroMartAgent
from .asr_normalizer import format_reply_for_ear
from .config import GNANI_DEFAULT_LANG, GNANI_DEFAULT_VOICE, GNANI_TTS_MODEL

# Map internal language codes to Gnani TTS/STT language codes (Page 11 of Capstone Brief)
GNANI_TTS_LANG_MAP = {
    "en": "en-IN",
    "en-IN": "en-IN",
    "hi": "hi-IN",
    "hi-IN": "hi-IN",
    "hi-en": "hi-en",
    "bn": "bn-IN",
    "gu": "gu-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "mr": "mr-IN",
    "pa": "pa-IN",
    "ta": "ta-IN",
    "te": "te-IN",
}


def get_gnani_clients():
    """Instantiate GnaniSTTClient and GnaniTTSClient from gnani-vachana SDK.

    Reads GNANI_API_KEY directly from environment variables.
    """
    from gnani.stt import GnaniSTTClient
    from gnani.tts import GnaniTTSClient

    stt = GnaniSTTClient()  # reads GNANI_API_KEY from environment
    tts = GnaniTTSClient()  # reads GNANI_API_KEY from environment
    return stt, tts


def voice_turn(
    audio_path: str,
    agent: ZentroMartAgent,
    language_code: str = GNANI_DEFAULT_LANG,
    voice: str = GNANI_DEFAULT_VOICE,
    model: str = GNANI_TTS_MODEL,
    output_file: str = "reply.wav",
) -> Tuple[str, str, Dict[str, Any]]:
    """Run one spoken turn: Gnani STT -> ZentroMartAgent -> Gnani TTS.

    Follows the official VoiceOps 2026 Step 4 specification:
    - Transcribes incoming audio via GnaniSTTClient.
    - Recovers gracefully if audio is empty or unclear.
    - Passes heard transcript to the Step 3 ZentroMartAgent.
    - Synthesizes speech via GnaniTTSClient with explicit model='timbre-v2.5' and voice='Kaveri'.
    """
    stt, tts = get_gnani_clients()
    stt_res = stt.transcribe(str(audio_path), language_code=language_code)
    heard = (stt_res.get("transcript") or "").strip() if isinstance(stt_res, dict) else str(stt_res).strip()

    if not heard:
        reply_text = "Sorry, I didn't catch that. Could you say it again?"
        result_meta = {
            "reply": reply_text,
            "decision": "clarify",
            "retrieved_chunks": [],
            "tool_calls": [],
            "language": "en",
        }
        tts_lang = language_code
    else:
        result_meta = agent.respond(heard, is_voice=True)
        reply_text = format_reply_for_ear(result_meta["reply"])
        result_meta["reply"] = reply_text
        detected_lang = result_meta.get("language", "en")
        tts_lang = GNANI_TTS_LANG_MAP.get(detected_lang, language_code)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    tts.synthesize(
        reply_text,
        voice=voice,
        model=model,
        language=tts_lang,
        output_file=str(output_file),
    )
    return heard, reply_text, result_meta
