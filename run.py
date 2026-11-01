#!/usr/bin/env python3
"""Main entry point for ZentroMart VoiceOps 2026 Support Agent.

Usage (Organizer Batch Audio Mode):
    python run.py <path_to_audio_folder> [--output-dir sample_outputs] [--language en-IN]

Usage (Practice Dataset Evaluation Mode):
    python run.py --evaluate [--dataset-dir /path/to/Capstone_Project_Dataset_FTTT_2026]

Usage (Interactive Web Interface):
    python run.py --web [--port 8000]
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from src.agent import ZentroMartAgent
from src.config import (
    GNANI_DEFAULT_LANG,
    GNANI_DEFAULT_VOICE,
    GNANI_TTS_MODEL,
    resolve_dataset_dir,
)
from src.evaluator import run_full_evaluation
from src.voice import voice_turn

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}


def _load_dotenv() -> None:
    """Load environment variables from .env if present (without requiring external packages)."""
    import os

    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _extract_conversation_key(audio_file: Path) -> Optional[str]:
    """Detect if an audio filename belongs to a multi-turn sequence (e.g., PC002_turn1.wav)."""
    m = re.match(r"^([A-Za-z0-9]+?)[_-](?:turn|t)\d+$", audio_file.stem, flags=re.IGNORECASE)
    return m.group(1) if m else None


def run_batch_audio(
    audio_dir: Path,
    output_dir: Path,
    dataset_dir: Path,
    language_code: str = GNANI_DEFAULT_LANG,
    voice: str = GNANI_DEFAULT_VOICE,
    model: str = GNANI_TTS_MODEL,
) -> List[Dict]:
    """Process a folder of customer audio files end-to-end with Gnani STT -> Agent -> Gnani TTS."""
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_files = sorted(
        [p for p in audio_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS]
    )

    if not audio_files:
        print(f"No audio files found in {audio_dir}. Running practice evaluation into {output_dir}...")
        run_full_evaluation(dataset_dir=dataset_dir, output_dir=output_dir)
        return []

    agent = ZentroMartAgent(dataset_dir=dataset_dir)
    current_conv_key: Optional[str] = None
    batch_records: List[Dict] = []

    for audio_path in audio_files:
        conv_key = _extract_conversation_key(audio_path)
        if conv_key is None or conv_key != current_conv_key:
            agent.reset()
            current_conv_key = conv_key

        stem = audio_path.stem
        out_wav = output_dir / f"{stem}_reply.wav"
        out_txt = output_dir / f"{stem}_reply.txt"
        out_src = output_dir / f"{stem}_sources.json"

        heard, reply_text, meta = voice_turn(
            audio_path=str(audio_path),
            agent=agent,
            language_code=language_code,
            voice=voice,
            model=model,
            output_file=str(out_wav),
        )

        out_txt.write_text(reply_text + "\n", encoding="utf-8")
        record = {
            "audio_file": audio_path.name,
            "transcript": heard,
            "reply_text": reply_text,
            "reply_audio": out_wav.name,
            "decision": meta.get("decision", "answer"),
            "retrieved_chunks": meta.get("retrieved_chunks", []),
            "tool_calls": meta.get("tool_calls", []),
            "language": meta.get("language", "en"),
        }
        out_src.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        batch_records.append(record)
        print(f"[OK] {audio_path.name} -> {out_wav.name} | chunks={record['retrieved_chunks']}")

    (output_dir / "batch_results.json").write_text(
        json.dumps(batch_records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "batch_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "audio_file",
                "transcript",
                "reply_text",
                "reply_audio",
                "decision",
                "retrieved_chunks",
                "tool_calls",
                "language",
            ],
        )
        writer.writeheader()
        for r in batch_records:
            writer.writerow({
                **r,
                "retrieved_chunks": " ; ".join(r["retrieved_chunks"]),
                "tool_calls": " ; ".join(r["tool_calls"]),
            })

    return batch_records


def main() -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(
        description="ZentroMart VoiceOps 2026 Customer Support Agent (Batch Audio, Evaluation, and Web UI)"
    )
    parser.add_argument(
        "audio_folder",
        nargs="?",
        default=None,
        help="Path to folder containing customer audio files (.wav/.mp3)",
    )
    parser.add_argument(
        "--input-dir",
        dest="input_dir",
        default=None,
        help="Optional flag alias for input audio folder",
    )
    parser.add_argument(
        "--output-dir",
        default="sample_outputs",
        help="Directory to save text replies, synthesized speech (.wav), and retrieved chunks",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Path to extracted Capstone_Project_Dataset_FTTT_2026 folder",
    )
    parser.add_argument(
        "--language",
        default=GNANI_DEFAULT_LANG,
        help="Gnani STT language code (e.g., en-IN, hi-IN, gu-IN, ta-IN)",
    )
    parser.add_argument(
        "--voice",
        default=GNANI_DEFAULT_VOICE,
        help="Gnani TTS voice name (e.g., Kaveri, Trupti, Pranav, Shlok)",
    )
    parser.add_argument(
        "--model",
        default=GNANI_TTS_MODEL,
        help="Gnani TTS model name (timbre-v2.5 or timbre-v3.0)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run Step 1, Step 2, Step 3, and Step 4 evaluation on practice/ CSVs",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch interactive browser-based VoiceOps support console",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the interactive web interface (default: 8000)",
    )

    args = parser.parse_args()
    dataset_dir = resolve_dataset_dir(args.dataset_dir)
    output_dir = Path(args.output_dir)

    if args.web:
        from src.web_app import run_web_server

        run_web_server(dataset_dir=dataset_dir, port=args.port)
        return 0

    target_folder = args.audio_folder or args.input_dir
    if args.evaluate or not target_folder:
        res = run_full_evaluation(dataset_dir=dataset_dir, output_dir=output_dir)
        print(json.dumps(res["summary"], indent=2))
        return 0

    audio_path = Path(target_folder)
    if not audio_path.exists():
        print(f"Error: Audio folder '{audio_path}' does not exist.", file=sys.stderr)
        return 1

    run_batch_audio(
        audio_dir=audio_path,
        output_dir=output_dir,
        dataset_dir=dataset_dir,
        language_code=args.language,
        voice=args.voice,
        model=args.model,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
