import json
import os


class BaseDashboard:
    def __init__(self):
        self.cache = {}

    def close(self):
        pass

    def log(self, data):
        self.cache.update(data)

    def log_table(self, name, columns=[], data=[]):
        self.cache[name] = [dict(zip(columns, row)) for row in data]

    def log_summary(self, key, value):
        if self.cache.get("summary") is None:
            self.cache["summary"] = {}
        self.cache["summary"][key] = value

    def log_summaries(self, data):
        if self.cache.get("summary") is None:
            self.cache["summary"] = {}
        self.cache["summary"].update(data)

    def save_json(self, file_dir):
        os.makedirs(file_dir, exist_ok=True)
        for key, value in self.cache.items():
            file_path = os.path.join(file_dir, f"{key}.json")
            with open(file_path, "w") as f:
                json.dump(value, f, ensure_ascii=False, indent=4)
