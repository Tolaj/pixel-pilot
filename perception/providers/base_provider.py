# perception/providers/base_provider.py
from ..base import PerceptionProvider


class LazyProvider(PerceptionProvider):
    """
    Mixin that adds lazy model loading to any provider.
    Subclasses implement _load() instead of worrying about init guards.
    """

    _loaded: bool = False

    def _load(self):
        """Override this to load models / open handles."""

    def _ensure_loaded(self):
        if not self._loaded:
            self._load()
            self._loaded = True

    def warm_up(self):
        self._ensure_loaded()
