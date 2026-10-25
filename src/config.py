"""Configuration and path discovery for the ZentroMart VoiceOps 2026 Agent."""
import os
from datetime import date
from pathlib import Path

# Fixed dataset reference date (KB-00.2)
REFERENCE_DATE = date(2026, 10, 20)
REFERENCE_DATE_STR = "2026-10-20"


def resolve_dataset_dir(explicit_dir: str = None) -> Path:
    """Locate the extracted Capstone_Project_Dataset_FTTT_2026 directory."""
    candidates = []
    if explicit_dir:
        candidates.append(Path(explicit_dir))
    env_dir = os.environ.get("ZENTROMART_DATASET_DIR") or os.environ.get("DATASET_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    here = Path(__file__).resolve().parent
    project_root = here.parent
    cwd = Path.cwd()

    for base in (cwd, project_root, project_root.parent, here):
        candidates.append(base)
        candidates.append(base / "Capstone_Project_Dataset_FTTT_2026")
        candidates.append(base / "work_dataset" / "Capstone_Project_Dataset_FTTT_2026")
        candidates.append(base.parent / "Capstone_Project_Dataset_FTTT_2026")
        candidates.append(base.parent / "work_dataset" / "Capstone_Project_Dataset_FTTT_2026")

    for cand in candidates:
        if (cand / "kb").is_dir() and (cand / "data").is_dir():
            return cand.resolve()

    return project_root.resolve()


# LLM & Voice settings from environment (.env)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "auto").lower()
LLM_MODEL = os.environ.get("LLM_MODEL", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

GNANI_API_KEY = os.environ.get("GNANI_API_KEY", "")
GNANI_TTS_MODEL = os.environ.get("GNANI_TTS_MODEL", "timbre-v2.5")
GNANI_DEFAULT_VOICE = os.environ.get("GNANI_VOICE", "Kaveri")
GNANI_DEFAULT_LANG = os.environ.get("GNANI_LANGUAGE", "en-IN")
