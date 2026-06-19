"""Canonical behaviour name map for BORIS labels across the 3 RS_Boris
projects. Maps each raw BORIS behaviour code to a canonical name used
by the PixelPaws Global Classifier Encyclopedia.

Behaviours with value `None` are EXCLUDED from v1 (per user direction).
"""

# raw BORIS code (case-sensitive) -> canonical name (or None to exclude)
ALIASES: dict[str, str | None] = {
    # ---- Flinching ----
    "L_flinching":       "L_flinching",
    "L flinching":       "L_flinching",
    "Left_flinching":    "L_flinching",
    "flinching":         "L_flinching",   # SNLT conf entry (no events yet)

    # ---- Licking (left only for v1; right paw not yet enough labels) ----
    "L_licking":         "Left_licking",
    "L licking":         "Left_licking",
    "Left_licking":      "Left_licking",
    "Licking":           "Left_licking",  # Rim project — pooled formalin paw
    "R licking":         None,
    "R_licking":         None,

    # ---- Scratching ----
    "Scratching":        "Scratching",

    # ---- Grooming family ----
    "Facial_grooming":   "Facial_grooming",
    "facial_grooming":   "Facial_grooming",
    "face_groom":        "Facial_grooming",
    "body_grooming":     "body_grooming",
    "body_groom":        "body_grooming",
    "back_groom":        "back_groom",
    "belly_groom":       "belly_groom",

    # ---- Postural / state ----
    "rearing":           "rearing",
    "walking":           "walking",
    "still":             "still",
    "unsupported_rear":  "unsupported_rear",   # 0 events in v1 — skip if empty

    # ---- Lifting (excluded for v1 — needs more labels) ----
    "L_lifting":         None,
    "L lifting":         None,
    "R_lifting":         None,
    "R lifting":         None,

    # ---- Shaking (excluded for v1 — only 50 + 2 events total) ----
    "L_shaking":         None,
    "L shaking":         None,
    "R_shaking":         None,
    "shaking":           None,
}


def canonical(raw: str) -> str | None:
    """Return canonical name for a raw BORIS behaviour code, or None
    if it's explicitly excluded from v1. Raises KeyError if the code
    isn't recognised — better to fail loud than silently drop labels."""
    if raw not in ALIASES:
        raise KeyError(
            f"Unknown BORIS behaviour code {raw!r}. Add to ALIASES in "
            f"scripts/research/boris_label_aliases.py before re-running."
        )
    return ALIASES[raw]
