import json

import hydra
from pydantic import BaseModel

from . import DatasetItem, DatasetItemForEvaluation


class MultiChallengeDatasetItemMixin(BaseModel):
    """Mixin class for MultiChallenge dataset item."""

    axis: str
    target_question: str
    requires_external_knowledge: bool | None


class MultiChallengeDatasetItem(DatasetItem, MultiChallengeDatasetItemMixin):
    """Dataset item for MultiChallenge dataset."""


class MultiChallengeDatasetItemForEvaluation(DatasetItemForEvaluation, MultiChallengeDatasetItemMixin):
    """Dataset item for MultiChallenge dataset for evaluation."""


def load_multichallenge(path: str) -> list[MultiChallengeDatasetItem]:
    path = hydra.utils.to_absolute_path(path)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for d in json.load(f):
            conversation = d["conversation"]
            assert len(conversation["role"]) == len(conversation["content"])

            prompt = []
            response = []
            for turn, (role, content) in enumerate(zip(conversation["role"], conversation["content"])):
                if turn % 2 == 0:
                    assert role == "user"
                    prompt.append(content)
                else:
                    assert role == "assistant"
                    response.append(content)

            assert role == "user"

            item = MultiChallengeDatasetItem(
                ID=d["question_id"],
                prompt=prompt,
                response=response,
                error_messages=[[]] * len(response),
                pattern=[None] * len(response),
                axis=d["axis"],
                target_question=d["target_question"],
                requires_external_knowledge=None,
            )
            data.append(item)

    return data


def load_multichallenge_ja(path: str) -> list[MultiChallengeDatasetItem]:
    path = hydra.utils.to_absolute_path(path)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for d in json.load(f):
            prompt = []
            response = []
            for turn, message in enumerate(d["CONVERSATION"]):
                if turn % 2 == 0:
                    assert message["role"] == "user"
                    prompt.append(message["content"])
                else:
                    assert message["role"] == "assistant"
                    response.append(message["content"])

            assert message["role"] == "user"

            item = MultiChallengeDatasetItem(
                ID=d["QUESTION_ID"],
                prompt=prompt,
                response=response,
                axis=d["AXIS"],
                target_question=d["TARGET_QUESTION"],
                requires_external_knowledge=d["REQUIRES_EXTERNAL_KNOWLEDGE"],
            )
            data.append(item)

    return data


def load_multichallenge_raw_output(path: str) -> list[MultiChallengeDatasetItem]:
    path = hydra.utils.to_absolute_path(path)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            item = MultiChallengeDatasetItem(**d)
            data.append(item)

    return data
