import os
import json


class BaseDashboard:
    def __init__(self):
        self.cache = {}

    def close(self):
        pass

    def log(self, data):
        self.cache.update(data)

    def log_summary(self, key, value):
        if self.cache.get("summary") is None:
            self.cache["summary"] = {}
        self.cache["summary"][key] = value

    def save_json(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(self.cache, f, indent=4, ensure_ascii=False)
