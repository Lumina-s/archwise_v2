from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from langchain_core.caches import BaseCache, RETURN_VAL_TYPE
from langchain_core.outputs import Generation


class LLMCache(BaseCache):
    def __init__(self, path: Path) -> None:
        self.path = path

    def get(self, namespace: str, payload: dict[str, Any]) -> Any | None:
        records = self._read_records()
        return records.get(self._key(namespace, payload))

    def set(self, namespace: str, payload: dict[str, Any], value: Any) -> None:
        records = self._read_records()
        records[self._key(namespace, payload)] = value
        self._write_records(records)

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        value = self.get("langchain", {"prompt": prompt, "llm_string": llm_string})
        if value is None:
            return None
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError("LangChain cache entry must be a list of strings.")
        return [Generation(text=item) for item in value]

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        texts = []
        for generation in return_val:
            if not isinstance(generation.text, str):
                raise ValueError("LangChain cache Generation text must be a string.")
            texts.append(generation.text)
        self.set("langchain", {"prompt": prompt, "llm_string": llm_string}, texts)

    def clear(self, **kwargs: Any) -> None:
        namespace = kwargs.get("namespace")
        if namespace is None:
            self._write_records({})
            return
        records = {
            key: value
            for key, value in self._read_records().items()
            if not key.startswith(f"{namespace}:")
        }
        self._write_records(records)

    def _read_records(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        with self.path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, dict):
            raise ValueError(f"LLM cache must be a JSON object: {self.path}")
        return records

    def _write_records(self, records: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, sort_keys=True)
        temp_path.replace(self.path)

    @staticmethod
    def _key(namespace: str, payload: dict[str, Any]) -> str:
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        return f"{namespace}:{digest}"
