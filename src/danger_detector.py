import re
import logging

def contains_danger_word(text: str, keywords: list) -> bool:
    """Check for whole-word matches (case-insensitive)."""
    if not text:
        return False
    text_lower = text.lower()
    for kw in keywords:
        pattern = r'\b' + re.escape(kw.lower()) + r'\b'
        if re.search(pattern, text_lower):
            logging.warning(f"🚨 DANGER WORD DETECTED: '{kw}' in '{text}'")
            return True
    return False