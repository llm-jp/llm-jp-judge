from collections.abc import MutableMapping, Sequence
from typing import TYPE_CHECKING, TypeVar, Union

from ..dataset import DatasetItem


if TYPE_CHECKING:
    from ..evaluator.base import BaseScoreExtractor

T = TypeVar("T", bound=DatasetItem)


class BaseClient:
    def __init__(
        self,
        model_name: str,
        max_retries: int = 1,
        async_request_interval: float = 1.0,
        disable_system_prompt: bool = False,
    ):
        self.model_name = model_name
        self.max_retries = max_retries
        self.async_request_interval = async_request_interval
        self.disable_system_prompt = disable_system_prompt

    def __call__(
        self,
        data: Sequence[T],
        score_extractor: Union["BaseScoreExtractor", None] = None,
        system_prompt: str | None = None,
        sampling_params: MutableMapping | None = None,
    ) -> Sequence[T]:
        raise NotImplementedError
