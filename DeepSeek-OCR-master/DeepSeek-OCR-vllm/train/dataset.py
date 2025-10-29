import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class Sample:
    image_path: str
    prompt: str
    response: str


class JsonlDataset:
    def __init__(self, jsonl_path: str):
        self.items: List[Sample] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.items.append(Sample(
                    image_path=obj["image_path"],
                    prompt=obj["prompt"],
                    response=obj["response"],
                ))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Sample:
        return self.items[idx]
