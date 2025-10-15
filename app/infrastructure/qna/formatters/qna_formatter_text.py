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
    include_confidence_bar: bool = False,
    include_confidence_stars: bool = False,
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
    _handle_parts_header(pkg=pkg, parts_list =parts, show_meta=show_meta)
    
    # -----------------------------------------
    # TOC
    # -----------------------------------------
    _handle_parts_toc(pkg=pkg, parts_list =parts, show_toc=show_toc)

    # -----------------------------------------
    # Q&A
    # -----------------------------------------
    _generate_parts_q_n_a(
        pkg=pkg,
        parts_list=parts,
        wrap_paragraph_width=width,
        include_confidence_bar=include_confidence_bar,
        include_confidence_stars=include_confidence_stars,
        ascii_bars=ascii_bars
    )

    # -----------------------------------------
    # DEBUG (optional)
    # -----------------------------------------
    if show_debug:
        _generate_parts_debug(pkg=pkg, parts_list=parts)

    return "\n".join(parts).rstrip() + "\n"

# ---------------------------------------------
# INTERNALS
# ---------------------------------------------

def _handle_parts_header(pkg:ProjectQnAPackage, parts_list: List[str], show_meta:bool):
    title = _safe(pkg.project_name) or "Project Q&A"
    parts_list.append(_rule("="))
    parts_list.append(f"PROJECT: {title}")
    parts_list.append(_rule("="))
    parts_list.append(f"REPO:      {_safe(pkg.repo_url)}")
    parts_list.append(f"REPO_ID:   {_safe(pkg.repo_id)}")
    
    if show_meta:
        if pkg.generated_at:
            dt = pkg.generated_at.astimezone(timezone.utc)
            parts_list.append(f"GENERATED: {dt.isoformat().replace('+00:00', 'Z')} (UTC)")
        if pkg.model:
            parts_list.append(f"MODEL:     {_safe(pkg.model)}")
        if pkg.prompt_version:
            parts_list.append(f"PROMPT_V:  {_safe(str(pkg.prompt_version))}")
        if pkg.prompt_tokens_hint is not None:
            parts_list.append(f"TOKENS:    {pkg.prompt_tokens_hint}")
    
    parts_list.append(_rule("-"))

def _handle_parts_toc(pkg:ProjectQnAPackage, parts_list: List[str], show_toc:bool):
    if show_toc and pkg.pairs:
        parts_list.append("CONTENTS:")
        for i, qa in enumerate(pkg.pairs, start=1):
            q_short = _one_line(qa.question, 80)
            parts_list.append(f"  {i}) {q_short}")
        parts_list.append(_rule("-"))

def _generate_parts_q_n_a(pkg:ProjectQnAPackage, parts_list: List[str], wrap_paragraph_width:int, include_confidence_bar:bool, include_confidence_stars:bool, ascii_bars:bool):
    for i, qa in enumerate(pkg.pairs, start=1):
        parts_list.append(f"[{i}] QUESTION")
        parts_list.append(_wrap(qa.question, wrap_paragraph_width, indent="  "))
        parts_list.append("STATUS")

        status_line = _status_line(
            qa,
            include_confidence_bar=include_confidence_bar,
            include_confidence_stars=include_confidence_stars,
            ascii_bars=ascii_bars,
        )
        parts_list.append(f"  {status_line}")

        parts_list.append("ANSWER")
        answer_text = (qa.answer or "").strip() or "(no answer)"
        parts_list.append(_wrap(answer_text, wrap_paragraph_width, indent="  "))

        if qa.evidence_snippets:
            parts_list.append("EVIDENCE")
            for s in _dedupe_preserve(qa.evidence_snippets):
                # bullet each evidence line and wrap
                bullet = f'- "{s.strip()}"'
                parts_list.append(_wrap(bullet, wrap_paragraph_width, indent="  "))

        parts_list.append(_rule("-"))

def _generate_parts_debug(pkg:ProjectQnAPackage, parts_list: List[str]):
    if pkg.raw_prompt:
        parts_list.append("DEBUG: RAW PROMPT")
        parts_list.append("<<<")
        parts_list.append(pkg.raw_prompt)
        parts_list.append(">>>")
        parts_list.append(_rule("-"))
    if pkg.raw_response:
        parts_list.append("DEBUG: RAW RESPONSE")
        parts_list.append("<<<")
        parts_list.append(pkg.raw_response)
        parts_list.append(">>>")
        parts_list.append(_rule("-"))

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
    
    bits: List[str] = [f"Status: {label}"]
    
    if qa.confidence is not None:
        bits.append(f"Confidence: {qa.confidence:.2f}")
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
