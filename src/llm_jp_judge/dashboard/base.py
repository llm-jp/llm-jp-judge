import json
import os
from typing import Any


class BaseDashboard:
    def __init__(self):
        self.cache = {}

    def close(self):
        pass

    def log(self, data: dict[str, Any]):
        self.cache.update(data)

    def log_table(self, name: str, columns: list[str] | None = None, data: list[list[Any]] | None = None):
        if columns is None:
            columns = []
        if data is None:
            data = []

        self.cache[name] = [dict(zip(columns, row)) for row in data]

    def log_summary(self, key: str, value: Any):
        if self.cache.get("summary") is None:
            self.cache["summary"] = {}
        self.cache["summary"][key] = value

    def log_summaries(self, data: dict[str, Any]):
        if self.cache.get("summary") is None:
            self.cache["summary"] = {}
        self.cache["summary"].update(data)

    def save_json(self, file_dir: str):
        os.makedirs(file_dir, exist_ok=True)
        for key, value in self.cache.items():
            file_path = os.path.join(file_dir, f"{key}.json")
            with open(file_path, "w") as f:
                json.dump(value, f, ensure_ascii=False, indent=4)
