import sqlite3
from abc import ABC, abstractmethod
from contextlib import closing
from pathlib import Path
from threading import Lock
from typing import Dict, Optional


class LangProCacheBackend(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        pass

    @abstractmethod
    def set(self, key: str, value: str) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


class InMemoryLangProCache(LangProCacheBackend):
    def __init__(self):
        self._data: Dict[str, str] = {}
        self._lock = Lock()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            return self._data.get(key)

    def set(self, key: str, value: str) -> None:
        with self._lock:
            self._data[key] = value

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class SQLiteLangProCache(LangProCacheBackend):
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path))
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA synchronous=NORMAL;")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS langpro_cache (
                    cache_key TEXT PRIMARY KEY,
                    response_text TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    "SELECT response_text FROM langpro_cache WHERE cache_key = ?",
                    (key,),
                ).fetchone()
        return None if row is None else row[0]

    def set(self, key: str, value: str) -> None:
        with self._lock:
            with closing(self._connect()) as connection:
                connection.execute(
                    """
                    INSERT INTO langpro_cache(cache_key, response_text)
                    VALUES(?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET response_text = excluded.response_text
                    """,
                    (key, value),
                )
                connection.commit()

    def clear(self) -> None:
        with self._lock:
            with closing(self._connect()) as connection:
                connection.execute("DELETE FROM langpro_cache")
                connection.commit()
