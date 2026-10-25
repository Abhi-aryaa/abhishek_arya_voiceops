"""ASR transcript normalization and speech output formatting for Step 4 VoiceOps."""
import re

SPOKEN_DIGIT_MAP = {
    "zero": "0", "oh": "0",
    "one": "1",
    "two": "2", "to": "2", "too": "2",
    "three": "3",
    "four": "4", "for": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def normalize_asr_text(text: str) -> str:
    """Normalize ASR transcript variations (spoken numbers, split acronyms, misheard product terms)."""
    if not text:
        return ""

    s = text.translate(DEVANAGARI_DIGITS).strip()

    # Strip leading conversational fillers common in ASR
    s = re.sub(r"^(?:uh+|um+|so|haan|hello|hi|hey)\s+", "", s, flags=re.IGNORECASE)

    # Fix common split words in ASR
    s = re.sub(r"\bu\s+p\s+i\b", "UPI", s, flags=re.IGNORECASE)
    s = re.sub(r"\bre\s+fund\b", "refund", s, flags=re.IGNORECASE)
    s = re.sub(r"\bkurtha\b", "kurta", s, flags=re.IGNORECASE)
    s = re.sub(r"\bis\s+the\s+free\b", "is the collection free", s, flags=re.IGNORECASE)

    # 1. Reconstruct spoken SKUs first: e.g. "ele-zero zero one to" -> "ELE-0012"
    digit_word_pat = r"(?:zero|oh|one|two|to|too|three|four|for|five|six|seven|eight|nine)"

    def _replace_sku(match: re.Match) -> str:
        prefix = match.group(1).upper()
        rest = match.group(2).lower().split()
        digits = []
        for tok in rest:
            if tok in SPOKEN_DIGIT_MAP:
                digits.append(SPOKEN_DIGIT_MAP[tok])
            elif tok.isdigit():
                digits.append(tok)
        joined = "".join(digits)
        if len(joined) == 4:
            return f"{prefix}-{joined}"
        return match.group(0)

    s = re.sub(
        rf"\b(ele|fsh|hom|bty|bks|gro|inr|dgt)[-\s]+({digit_word_pat}(?:\s+{digit_word_pat}){{3}})\b",
        _replace_sku,
        s,
        flags=re.IGNORECASE,
    )

    # 2. Reconstruct spoken multi-digit numbers (years/capacities) like "two zero two six" -> "2026"
    def _replace_spoken_digits(match: re.Match) -> str:
        tokens = match.group(0).lower().split()
        return "".join(SPOKEN_DIGIT_MAP.get(t, t) for t in tokens)

    s = re.sub(
        rf"\b{digit_word_pat}(?:\s+{digit_word_pat}){{2,5}}\b",
        _replace_spoken_digits,
        s,
        flags=re.IGNORECASE,
    )

    # Fix "10000 mah" -> "10000 mAh" and "256 gb" -> "256 GB"
    s = re.sub(r"\b10000\s*mah\b", "10000 mAh", s, flags=re.IGNORECASE)
    s = re.sub(r"\b20000\s*mah\b", "20000 mAh", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(\d{2,3})\s*gb\b", r"\1 GB", s, flags=re.IGNORECASE)

    # Fix single spoken numbers before time units (e.g., "ten days" -> "10 days")
    word_num_map = {
        "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
        "fifteen": "15", "thirty": "30", "forty-eight": "48", "forty eight": "48",
    }
    for word, num in word_num_map.items():
        s = re.sub(rf"\b{word}\s+(days?|hours?|minutes?|seconds?|months?)\b", rf"{num} \1", s, flags=re.IGNORECASE)

    return s.strip()


def format_reply_for_ear(reply: str) -> str:
    """Format text replies for natural TTS synthesis (STY-3 & Step 4 brief).

    - Breaks 8-digit order numbers into groups of four digits with spaces (e.g. '8978 2837').
    - Keeps sentences concise and clear for speech synthesis.
    """
    if not reply:
        return ""
    formatted = re.sub(r"(?<!\d)(\d{4})(\d{4})(?!\d)", r"\1 \2", reply)
    return formatted
