"""Utility functions for CivicLens."""


def validate_complaint(complaint: str) -> bool:
    """Check whether a complaint contains meaningful text."""
    return bool(complaint and complaint.strip())


def build_complaint_prompt(complaint: str) -> str:
    """Create an instruction prompt for civic issue analysis."""
    return f"""
Analyze this civic complaint:

{complaint.strip()}

Return:
1. Issue category
2. Priority: Low, Medium, or High
3. Short summary
4. Recommended civic action

Base your response only on the complaint details.
"""


def normalize_priority(priority: str) -> str:
    """Normalize a priority label."""
    value = priority.strip().lower()
    if value in {"high", "medium", "low"}:
        return value.title()
    return "Medium"
