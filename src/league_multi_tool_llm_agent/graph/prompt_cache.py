from typing import Protocol


class PromptCache(Protocol):
    def insert(self, key: str, value: str) -> None: ...

    def get(self, key: str) -> str | None: ...


class InMemoryDictCache:
    def __init__(self) -> None:
        self.cache: dict = {}

    def insert(self, key: str, value: str) -> None:
        self.cache[key] = value

    def get(self, key: str) -> str | None:
        return self.cache.get(key, None)
