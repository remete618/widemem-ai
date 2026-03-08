from __future__ import annotations

from abc import ABC, abstractmethod

from widemem.core.types import Fact


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, text: str) -> list[Fact]:
        ...
