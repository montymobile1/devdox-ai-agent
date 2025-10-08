# ---------------------------------------------
# QNA PLAIN TEXT FORMATTER
# ---------------------------------------------
from __future__ import annotations

from typing import Iterable, List, Optional
from datetime import timezone
import textwrap

from app.infrastructure.qna.qna_models import ProjectQnAPackage, QAPair


# ---------------------------------------------
# PUBLIC API
# ---------------------------------------------
def format_qna_text(
    pkg: ProjectQnAPackage,
    *,
    width: int = 88,
    show_meta: bool = True,
    show_toc: bool = True,
    show_debug: bool = False,      # append raw prompt/response at the end
    include_confidence_bar: bool = True,
    include_confidence_stars: bool = True,
    ascii_bars: bool = False,      # use "#"/"." instead of unicode blocks
) -> str:
    """
    Build a plain-text (no markdown) report from a ProjectQnAPackage.

    Parameters
    ----------
    width : int
        Target wrap width for paragraphs.
    show_meta : bool
        Include repo id, timestamp, model, etc.
    show_toc : bool
        Include numbered list of questions up-front.
    show_debug : bool
        Append DEBUG sections containing raw prompt/response.
    include_confidence_bar : bool
        Show a 10-step bar for confidence.
    include_confidence_stars : bool
        Show a 0–5 star rendering of confidence.
    ascii_bars : bool
        If True, render confidence bar as '#' and '.' (terminal-safe everywhere).

    Returns
    -------
    str
        Pure text string suitable for API responses and plain inspectors.
    """
    parts: List[str] = []

    # -----------------------------------------
    # Header
    # -----------------------------------------
    title = _safe(pkg.project_name) or "Project Q&A"
    parts.append(_rule("="))
    parts.append(f"PROJECT: {title}")
    parts.append(_rule("="))
    parts.append(f"REPO:      {_safe(pkg.repo_url)}")
    parts.append(f"REPO_ID:   {_safe(pkg.repo_id)}")

    if show_meta:
        if pkg.generated_at:
            dt = pkg.generated_at.astimezone(timezone.utc)
            parts.append(f"GENERATED: {dt.isoformat().replace('+00:00', 'Z')} (UTC)")
        if pkg.model:
            parts.append(f"MODEL:     {_safe(pkg.model)}")
        if pkg.prompt_version:
            parts.append(f"PROMPT_V:  {_safe(str(pkg.prompt_version))}")
        if pkg.prompt_tokens_hint is not None:
            parts.append(f"TOKENS:    {pkg.prompt_tokens_hint}")

    parts.append(_rule("-"))

    # -----------------------------------------
    # TOC
    # -----------------------------------------
    if show_toc and pkg.pairs:
        parts.append("CONTENTS:")
        for i, qa in enumerate(pkg.pairs, start=1):
            q_short = _one_line(qa.question, 80)
            parts.append(f"  {i}) {q_short}")
        parts.append(_rule("-"))

    # -----------------------------------------
    # Q&A
    # -----------------------------------------
    for i, qa in enumerate(pkg.pairs, start=1):
        parts.append(f"[{i}] QUESTION")
        parts.append(_wrap(qa.question, width, indent="  "))
        parts.append("STATUS")

        status_line = _status_line(
            qa,
            include_confidence_bar=include_confidence_bar,
            include_confidence_stars=include_confidence_stars,
            ascii_bars=ascii_bars,
        )
        parts.append(f"  {status_line}")

        parts.append("ANSWER")
        answer_text = (qa.answer or "").strip() or "(no answer)"
        parts.append(_wrap(answer_text, width, indent="  "))

        if qa.evidence_snippets:
            parts.append("EVIDENCE")
            for s in _dedupe_preserve(qa.evidence_snippets):
                # bullet each evidence line and wrap
                bullet = f'- "{s.strip()}"'
                parts.append(_wrap(bullet, width, indent="  "))

        parts.append(_rule("-"))

    # -----------------------------------------
    # DEBUG (optional)
    # -----------------------------------------
    if show_debug:
        if pkg.raw_prompt:
            parts.append("DEBUG: RAW PROMPT")
            parts.append("<<<")
            parts.append(pkg.raw_prompt)
            parts.append(">>>")
            parts.append(_rule("-"))
        if pkg.raw_response:
            parts.append("DEBUG: RAW RESPONSE")
            parts.append("<<<")
            parts.append(pkg.raw_response)
            parts.append(">>>")
            parts.append(_rule("-"))

    return "\n".join(parts).rstrip() + "\n"


# ---------------------------------------------
# INTERNAL HELPERS
# ---------------------------------------------
def _safe(text: Optional[str]) -> str:
    return "" if text is None else str(text)

def _one_line(text: str, max_len: int) -> str:
    t = " ".join(text.split())
    return t if len(t) <= max_len else t[: max_len - 1] + "…"

def _rule(ch: str, width: int = 88) -> str:
    ch = ch[:1] or "-"
    return ch * width

def _wrap(text: str, width: int, indent: str = "") -> str:
    text = (text or "").replace("\r\n", "\n").strip()
    if not text:
        return indent
    wrapper = textwrap.TextWrapper(
        width=width,
        initial_indent=indent,
        subsequent_indent=indent,
        drop_whitespace=True,
        replace_whitespace=True,
    )
    return wrapper.fill(" ".join(text.split()))

def _confidence_bar(conf: float, width: int = 10, ascii_bars: bool = False) -> str:
    conf = max(0.0, min(1.0, float(conf or 0.0)))
    filled = int(round(conf * width))
    empty = width - filled
    if ascii_bars:
        return "#" * filled + "." * empty
    # Unicode blocks render nicely in modern terminals/inspectors
    return "█" * filled + "░" * empty

def _confidence_stars(conf: float, total: int = 5) -> str:
    conf = max(0.0, min(1.0, float(conf or 0.0)))
    filled = int(round(conf * total))
    return "★" * filled + "☆" * (total - filled)

def _status_line(
    qa: QAPair,
    *,
    include_confidence_bar: bool,
    include_confidence_stars: bool,
    ascii_bars: bool,
) -> str:
    if qa.insufficient_evidence:
        label = "Insufficient evidence"
    elif (qa.answer or "").strip().lower().startswith("inferred:"):
        label = "Inferred"
    else:
        label = "Evidence-backed"

    bits: List[str] = [f"Status: {label}", f"Confidence: {qa.confidence:.2f}"]
    if include_confidence_stars:
        bits.append(f"Stars: {_confidence_stars(qa.confidence)}")
    if include_confidence_bar:
        bits.append(f"Bar: {_confidence_bar(qa.confidence, ascii_bars=ascii_bars)}")
    return " | ".join(bits)

def _dedupe_preserve(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for it in items:
        key = (it or "").strip()
        if key and key not in seen:
            seen.add(key)
            out.append(it)
    return out
