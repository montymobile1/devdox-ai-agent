from __future__ import annotations

import inspect
from typing import Callable, Dict, List, Tuple


class CallSpyMixin:
    """Collects (method_name, args, kwargs) for assertions."""

    def __init__(self) -> None:
        self.received_calls: List[Tuple[str, tuple, dict]] = []

    def _name(self, method: Callable) -> str:
        return method.__name__

    def _touch(self, method: Callable, /, *args, **kwargs) -> str:
        name = self._name(method)
        self.received_calls.append((name, args, kwargs))
        return name


class ExceptionPlanMixin:
    """Lets you pre-wire exceptions per method name."""

    def __init__(self) -> None:
        self._exceptions: Dict[str, Exception] = {}

    def set_exception(self, method: Callable, exc: Exception) -> None:
        self._exceptions[method.__name__] = exc

    def _maybe_raise(self, method_name: str) -> None:
        exc = self._exceptions.get(method_name)
        if exc:
            raise exc

class FakeBase(CallSpyMixin, ExceptionPlanMixin):
    """
    For fakes: just call tracking + exceptions.
    Your class keeps its own store and accessors.
    """

    def __init__(self) -> None:
        CallSpyMixin.__init__(self)
        ExceptionPlanMixin.__init__(self)

    def _before(self, method: Callable, /, *args, **kwargs) -> str:
        mname = self._touch(method, *args, **kwargs)
        self._maybe_raise(mname)
        return mname
