"""Date and business-day calculations grounded in KB-00.2, KB-00.3, and data/holidays.csv."""
import csv
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional, Set, Tuple

from .config import REFERENCE_DATE, resolve_dataset_dir

MONTH_MAP = {
    "january": 1, "jan": 1, "जनवरी": 1,
    "february": 2, "feb": 2, "फ़रवरी": 2, "फरवरी": 2,
    "march": 3, "mar": 3, "मार्च": 3,
    "april": 4, "apr": 4, "अप्रैल": 4,
    "may": 5, "मई": 5,
    "june": 6, "jun": 6, "जून": 6,
    "july": 7, "jul": 7, "जुलाई": 7,
    "august": 8, "aug": 8, "अगस्त": 8,
    "september": 9, "sep": 9, "sept": 9, "सितंबर": 9, "सितम्बर": 9,
    "october": 10, "oct": 10, "अक्टूबर": 10,
    "november": 11, "nov": 11, "नवंबर": 11, "नवम्बर": 11,
    "december": 12, "dec": 12, "दिसंबर": 12, "दिसम्बर": 12,
}

EN_MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

HI_MONTH_NAMES = [
    "", "जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून",
    "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"
]


def load_holidays(dataset_dir: Optional[Path] = None) -> Set[date]:
    """Load warehouse holiday dates from data/holidays.csv."""
    base = dataset_dir or resolve_dataset_dir()
    holidays_file = base / "data" / "holidays.csv"
    holidays = set()
    if holidays_file.is_file():
        with holidays_file.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                d_str = (row.get("date") or "").strip()
                if d_str:
                    holidays.add(date.fromisoformat(d_str[:10]))
    else:
        # Fallback from KB-00.3 table
        for d_str in ("2026-08-15", "2026-10-02", "2026-10-19", "2026-11-09"):
            holidays.add(date.fromisoformat(d_str))
    return holidays


def is_business_day(d: date, holidays: Set[date]) -> bool:
    """Business days are Monday to Saturday (weekday < 6), excluding warehouse holidays."""
    return d.weekday() < 6 and d not in holidays


def add_business_days(start_date: date, num_days: int, holidays: Optional[Set[date]] = None) -> date:
    """Add `num_days` business days to `start_date` (KB-00.3)."""
    if holidays is None:
        holidays = load_holidays()
    current = start_date
    added = 0
    while added < num_days:
        current += timedelta(days=1)
        if is_business_day(current, holidays):
            added += 1
    return current


def business_days_between(start_date: date, end_date: date, holidays: Optional[Set[date]] = None) -> int:
    """Count business days elapsed after `start_date` up to `end_date`."""
    if holidays is None:
        holidays = load_holidays()
    if end_date <= start_date:
        return 0
    current = start_date
    count = 0
    while current < end_date:
        current += timedelta(days=1)
        if is_business_day(current, holidays):
            count += 1
    return count


def add_calendar_days(start_date: date, num_days: int) -> date:
    """Add calendar days for return window calculations (RET-2)."""
    return start_date + timedelta(days=num_days)


def parse_iso_date(val: Optional[str]) -> Optional[date]:
    """Parse YYYY-MM-DD or YYYY-MM-DDTHH:MM string into a date object."""
    if not val or not str(val).strip():
        return None
    s = str(val).strip()[:10]
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def extract_dates_from_text(text: str, ref_date: date = REFERENCE_DATE) -> List[date]:
    """Extract explicit or relative dates from a customer message."""
    found: List[date] = []
    lower = text.lower()

    # Check ISO dates first: YYYY-MM-DD
    for m in re.finditer(r"\b(202\d)-(\d{1,2})-(\d{1,2})\b", lower):
        try:
            found.append(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            pass

    # Check DD Month [YYYY]
    month_pattern = "|".join(re.escape(k) for k in sorted(MONTH_MAP.keys(), key=len, reverse=True))
    pattern = rf"\b(\d{{1,2}})\s+({month_pattern})(?:\s+(202\d))?\b"
    for m in re.finditer(pattern, lower):
        day = int(m.group(1))
        month = MONTH_MAP[m.group(2)]
        year = int(m.group(3)) if m.group(3) else ref_date.year
        try:
            d = date(year, month, day)
            if d not in found:
                found.append(d)
        except ValueError:
            pass

    # Check relative terms
    if re.search(r"\b(yesterday|kal)\b", lower) or "कल" in text:
        d = ref_date - timedelta(days=1)
        if d not in found:
            found.append(d)
    if re.search(r"\b(day before yesterday|parso|parson)\b", lower) or "परसों" in text:
        d = ref_date - timedelta(days=2)
        if d not in found:
            found.append(d)
    if re.search(r"\b(today|aaj)\b", lower) or "आज" in text:
        if ref_date not in found:
            found.append(ref_date)

    return found


def format_date_human(d: date, lang: str = "en") -> str:
    """Format a date with day, month name, and year (e.g., '27 October 2026' or '27 अक्टूबर 2026')."""
    if lang == "hi":
        return f"{d.day} {HI_MONTH_NAMES[d.month]} {d.year}"
    return f"{d.day} {EN_MONTH_NAMES[d.month]} {d.year}"
