"""
Text normalization, tokenization, script detection, and numeric token extraction.
"""

import re
import unicodedata
from typing import List, Set

# Common legal and generic entity terms to strip for core name extraction
LEGAL_TERMS: Set[str] = {
    "pvt", "ltd", "private", "limited", "inc", "incorporated", "corp", "corporation",
    "llc", "llp", "gmbh", "sa", "sarl", "co", "company", "group", "holdings",
    "services", "solutions", "enterprise", "enterprises", "traders", "trading",
    "p", "l", "nv", "bv", "plc"
}

# Regex for non-alphanumeric unicode characters (keeps letters, marks, digits in all scripts)
NON_ALPHANUMERIC_RE = re.compile(r"[^\w\s]", re.UNICODE)
DIGITS_RE = re.compile(r"\b\d+\b")


def normalize_text(text: str) -> str:
    """
    Normalize text: NFKD unicode normalization, lowercase, strip punctuation, strip extra whitespace.
    Handles multi-script text safely.
    """
    if not text or not isinstance(text, str):
        return ""

    # NFKC normalization standardizes combined unicode characters
    norm = unicodedata.normalize("NFKC", text).lower()
    # Replace non-alphanumeric characters with spaces
    cleaned = NON_ALPHANUMERIC_RE.sub(" ", norm)
    # Collapse multiple whitespace characters
    return " ".join(cleaned.split())


def extract_core_name(normalized_name: str) -> str:
    """
    Remove standard legal suffixes/words from normalized name.
    Example: 'raj investments pvt ltd' -> 'raj investments'
    """
    if not normalized_name:
        return ""

    tokens = normalized_name.split()
    core_tokens = [t for t in tokens if t not in LEGAL_TERMS]
    return " ".join(core_tokens) if core_tokens else normalized_name


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens."""
    norm = normalize_text(text)
    return norm.split() if norm else []


def extract_numeric_tokens(text: str) -> List[str]:
    """
    Extract numeric tokens from text.
    Example: 'Door 12-34 Street 5' -> ['12', '34', '5']
    """
    if not text or not isinstance(text, str):
        return []

    # Find digit sequences in original or normalized text
    matches = DIGITS_RE.findall(text)
    return matches


def detect_script(text: str) -> str:
    """
    Detect the primary script of a string using Unicode block definitions.
    Supported script classes: Latin, Devanagari, Telugu, Tamil, Kannada, Bengali, Gujarati, Malayalam.
    """
    if not text or not isinstance(text, str):
        return "Unknown"

    script_counts = {
        "Latin": 0,
        "Devanagari": 0,
        "Telugu": 0,
        "Tamil": 0,
        "Kannada": 0,
        "Bengali": 0,
        "Gujarati": 0,
        "Malayalam": 0,
        "Other": 0,
    }

    for char in text:
        if not char.isalpha():
            continue
        try:
            name = unicodedata.name(char, "")
            if "LATIN" in name:
                script_counts["Latin"] += 1
            elif "DEVANAGARI" in name:
                script_counts["Devanagari"] += 1
            elif "TELUGU" in name:
                script_counts["Telugu"] += 1
            elif "TAMIL" in name:
                script_counts["Tamil"] += 1
            elif "KANNADA" in name:
                script_counts["Kannada"] += 1
            elif "BENGALI" in name:
                script_counts["Bengali"] += 1
            elif "GUJARATI" in name:
                script_counts["Gujarati"] += 1
            elif "MALAYALAM" in name:
                script_counts["Malayalam"] += 1
            else:
                script_counts["Other"] += 1
        except ValueError:
            script_counts["Other"] += 1

    # Return script with maximum count (excluding 0)
    max_script = max(script_counts.items(), key=lambda x: x[1])
    if max_script[1] == 0:
        return "Latin"  # Default fallback if no alpha characters found
    return max_script[0]
