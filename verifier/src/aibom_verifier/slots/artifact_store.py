from pathlib import Path
from typing import Protocol


def _sanitize_key(key: str) -> str:
    return key.replace("/", "__")


class ArtifactStore(Protocol):
    def exists(self, key: str) -> bool: ...

    def get(self, key: str) -> bytes | None: ...

    def put(self, key: str, data: bytes) -> None: ...


class FilesystemArtifactStore:
    def __init__(self, base_dir: Path, ignore_cache: bool = False) -> None:
        self._base_dir = base_dir
        self._ignore_cache = ignore_cache
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        return self._base_dir / _sanitize_key(key)

    def exists(self, key: str) -> bool:
        if self._ignore_cache:
            return False
        return self._path_for(key).is_file()

    def get(self, key: str) -> bytes | None:
        if self._ignore_cache:
            return None
        path = self._path_for(key)
        if not path.is_file():
            return None
        return path.read_bytes()

    def put(self, key: str, data: bytes) -> None:
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


class InMemoryArtifactStore:
    def __init__(self, ignore_cache: bool = False) -> None:
        self._ignore_cache = ignore_cache
        self._data: dict[str, bytes] = {}

    def exists(self, key: str) -> bool:
        if self._ignore_cache:
            return False
        return key in self._data

    def get(self, key: str) -> bytes | None:
        if self._ignore_cache:
            return None
        return self._data.get(key)

    def put(self, key: str, data: bytes) -> None:
        self._data[key] = data


class CountingArtifactStore:
    def __init__(self, inner: ArtifactStore) -> None:
        self._inner = inner
        self.hits: list[str] = []
        self.misses: list[str] = []

    def exists(self, key: str) -> bool:
        if self._inner.exists(key):
            self.hits.append(key)
            return True
        self.misses.append(key)
        return False

    def get(self, key: str) -> bytes | None:
        data = self._inner.get(key)
        if data is not None:
            self.hits.append(key)
        else:
            self.misses.append(key)
        return data

    def put(self, key: str, data: bytes) -> None:
        self._inner.put(key, data)
