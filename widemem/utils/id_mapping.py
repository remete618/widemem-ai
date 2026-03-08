from __future__ import annotations


class IDMapper:
    """Maps UUIDs to simple integers for LLM calls, preventing hallucinated IDs."""

    def __init__(self) -> None:
        self._uuid_to_int: dict[str, int] = {}
        self._int_to_uuid: dict[int, str] = {}
        self._next = 1

    def add(self, uuid: str) -> int:
        if uuid in self._uuid_to_int:
            return self._uuid_to_int[uuid]
        idx = self._next
        self._uuid_to_int[uuid] = idx
        self._int_to_uuid[idx] = uuid
        self._next += 1
        return idx

    def to_uuid(self, idx: int) -> str | None:
        return self._int_to_uuid.get(idx)

    def to_int(self, uuid: str) -> int | None:
        return self._uuid_to_int.get(uuid)
