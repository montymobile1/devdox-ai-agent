from __future__ import annotations
import os
import traceback
from typing import List, Dict, Any, Tuple


def _exc_message(te: traceback.TracebackException) -> str:
    """
    Extract message from TracebackException.
    "ZeroDivisionError: division by zero" -> "division by zero"
    """
    exc_only = "".join(te.format_exception_only()).strip()
    if ": " in exc_only:
        return exc_only.split(": ", 1)[1].strip()
    return exc_only


def _next_link(tb: traceback.TracebackException) -> traceback.TracebackException | None:
    """Prefer explicit causes; fall back to context (unless suppressed)."""
    if tb.__cause__ is not None:
        return tb.__cause__
    if tb.__suppress_context__:
        return None
    return tb.__context__

def _truncate(msg: str, limit: int) -> str:
    return (msg if not limit or len(msg) <= limit else f"{msg[:limit-1]}…")

def _node_from_tbexc(
    tb: traceback.TracebackException,
    include_location: bool,
    msg_limit: int,
) -> Dict[str, Any]:
    frame = tb.stack[-1] if tb.stack else None

    func = frame.name if frame else "<unknown>"
    if include_location and frame:
        filename = os.path.basename(frame.filename)
        lineno = frame.lineno
    else:
        filename = None
        lineno = None

    exc_type = getattr(tb, "exc_type", None)
    type_name = getattr(exc_type, "__name__", getattr(tb, "exc_type_name", "Exception"))
    msg = _truncate(_exc_message(tb), msg_limit)

    return {
        "func": func,
        "type": type_name,
        "msg": msg,
        "file": filename,
        "line": lineno,
    }


def build_error_chain_for_template(
    exc: BaseException,
    *,
    include_location: bool = True,
    include_locals: bool = False,
    msg_limit: int = 200,
) -> List[Dict[str, Any]]:
    """
    Returns OUTER→INNER nodes for top-down display.
    Each node: {depth, func, type, msg, file, line}
    """
    tb = traceback.TracebackException.from_exception(exc, capture_locals=include_locals)
    raw: List[Dict[str, Any]] = []

    while tb:
        raw.append(_node_from_tbexc(tb, include_location, msg_limit))
        tb = _next_link(tb)

    # Reverse to OUTER→INNER and assign depth in one expression (no extra loop body)
    return [{**node, "depth": i} for i, node in enumerate(reversed(raw))]


def make_plain_stacktrace(
    exc: BaseException, *, max_chars: int = 16000
) -> Tuple[str, bool]:
    """
    Returns (stacktrace_text, truncated_flag).
    """
    text = "".join(traceback.format_exception(exc))
    truncated = False
    if max_chars and len(text) > max_chars:
        text = text[: max_chars - 1] + "…"
        truncated = True
    return text, truncated
