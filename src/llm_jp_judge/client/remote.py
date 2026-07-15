import asyncio
import logging
import warnings
from collections.abc import MutableMapping, Sequence
from copy import deepcopy
from typing import TypeVar, cast

import openai
import tqdm
import tqdm.asyncio
from anthropic import AnthropicBedrock as AnthropicBedrockClient
from anthropic.types import Message, MessageParam, TextBlock
from dotenv import load_dotenv
from openai import AzureOpenAI as AzureOpenAIClient
from openai import OpenAI as OpenAIClient

from ..dataset import DatasetItem
from ..evaluator.base import BaseScoreExtractor
from .base import BaseClient


T = TypeVar("T", bound=DatasetItem)


load_dotenv(override=True)


class OpenAI(BaseClient):
    def __init__(
        self,
        model_name: str = "gpt-4o-2024-08-06",
        max_retries: int = 1,
        async_request_interval: float = 1.0,
        disable_system_prompt: bool = False,
        api_key: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        base_url: str | None = None,
    ):
        self.model_name = model_name
        self.max_retries = max_retries
        self.async_request_interval = async_request_interval
        self.disable_system_prompt = disable_system_prompt

        self.client = OpenAIClient(
            api_key=api_key,
            organization=organization,
            project=project,
            base_url=base_url,
        )

    def get_messages(
        self,
        prompt: list[str],
        response: list[str | None],
        system_prompt: str | None = None,
    ) -> list[dict[str, str | None]]:
        if self.disable_system_prompt and system_prompt is not None:
            prompt = deepcopy(prompt)
            prompt[0] = f"{system_prompt}\n\n{prompt[0]}"
            system_prompt = None

        messages: list[dict[str, str | None]] = []
        for turn in range(len(prompt)):
            messages.append({"role": "user", "content": prompt[turn]})
            if turn < len(response):
                messages.append({"role": "assistant", "content": response[turn]})

        if system_prompt is not None:
            messages.insert(0, {"role": "system", "content": system_prompt})

        return messages

    async def async_request(
        self,
        prompt: list[str],
        response: list[str | None],
        system_prompt: str | None = None,
        sampling_params: MutableMapping | None = None,
    ) -> str | None:
        if sampling_params is None:
            sampling_params = {}

        messages = await asyncio.to_thread(self.get_messages, prompt, response, system_prompt=system_prompt)

        client_response = await asyncio.to_thread(
            self.client.chat.completions.create,  # type: ignore[arg-type]
            model=self.model_name,
            messages=messages,
            **sampling_params,
        )
        return client_response.choices[0].message.content

    async def process_data(
        self,
        data: Sequence[T],
        score_extractor: BaseScoreExtractor | None = None,
        system_prompt: str | None = None,
        sampling_params: MutableMapping | None = None,
    ) -> Sequence[T]:
        if sampling_params is None:
            sampling_params = {}

        tasks = []
        wait = 0.0
        for d in data:
            tasks.append(
                self._process_single_request(
                    d,
                    score_extractor,
                    system_prompt,
                    wait=wait,
                    sampling_params=sampling_params,
                )
            )
            wait += self.async_request_interval * len(d.prompt)

        data = await tqdm.asyncio.tqdm.gather(*tasks, desc=self.model_name)

        return data

    async def _process_single_request(
        self,
        d: T,
        score_extractor: BaseScoreExtractor | None,
        system_prompt: str | None,
        wait: float,
        sampling_params: MutableMapping | None = None,
    ) -> T:
        if sampling_params is None:
            sampling_params = {}

        await asyncio.sleep(wait)

        d.response, d.pattern, d.error_messages = [], [], []
        for turn in range(len(d.prompt)):
            retry_count = 0
            sleep = 0.0

            d.response.append(None)
            d.pattern.append(None)
            d.error_messages.append([])
            while retry_count <= self.max_retries:
                if len(d.error_messages[-1]) > 0:
                    logging.warning(f"{d.error_messages[-1][-1]}. Retrying in {sleep} seconds.")
                await asyncio.sleep(sleep)

                try:
                    d.response[-1] = await self.async_request(
                        d.prompt[: turn + 1],
                        d.response[:turn],
                        system_prompt=system_prompt,
                        sampling_params=sampling_params,
                    )
                except (openai.RateLimitError, openai.APITimeoutError) as e:
                    d.error_messages[-1].append(str(e))
                    sleep = 60
                except openai.BadRequestError as e:
                    d.error_messages[-1].append(str(e))

                    retry_count += 1
                    sleep = self.async_request_interval
                    await asyncio.sleep(self.async_request_interval)
                else:
                    if score_extractor is not None:
                        try:
                            assert d.response[-1] is not None
                            d.pattern[-1] = score_extractor(d.response[-1])
                        except Exception as e:
                            d.error_messages[-1].append(str(e))
                            retry_count += 1
                            sleep = self.async_request_interval
                            continue
                    break

            if turn < len(d.prompt) - 1:
                await asyncio.sleep(self.async_request_interval)

        return d

    def fill_sampling_params(self, sampling_params: MutableMapping) -> MutableMapping:
        return {k: v for k, v in sampling_params.items() if v is not None}

    def update_sampling_params(self, sampling_params: MutableMapping) -> MutableMapping:
        if self.model_name.startswith("gpt-5"):
            if "max_tokens" in sampling_params:
                sampling_params["max_completion_tokens"] = sampling_params.pop("max_tokens")
            if "top_p" in sampling_params:
                logging.warning("gpt-5 does not support top_p parameter. Ignoring.")
                del sampling_params["top_p"]
        return sampling_params

    def __call__(
        self,
        data: Sequence[T],
        score_extractor: BaseScoreExtractor | None = None,
        system_prompt: str | None = None,
        sampling_params: MutableMapping | None = None,
    ) -> Sequence[T]:
        if sampling_params is None:
            sampling_params = {}

        sampling_params = self.fill_sampling_params(sampling_params)
        sampling_params = self.update_sampling_params(sampling_params)

        return asyncio.run(self.process_data(data, score_extractor, system_prompt, sampling_params=sampling_params))


class AzureOpenAI(OpenAI):
    def __init__(
        self,
        model_name: str = "gpt-4o-2024-08-06",
        max_retries: int = 1,
        async_request_interval: float = 1.0,
        disable_system_prompt: bool = False,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
    ):
        self.model_name = model_name
        self.max_retries = max_retries
        self.async_request_interval = async_request_interval
        self.disable_system_prompt = disable_system_prompt

        self.client = AzureOpenAIClient(
            azure_endpoint=azure_endpoint,  # type: ignore[arg-type]
            api_version=api_version,
            api_key=api_key,
        )


class BedrockAnthropic(AzureOpenAI):
    def __init__(
        self,
        model_name: str = "anthropic.claude-3-5-sonnet-20240620-v1:0",
        max_retries: int = 1,
        async_request_interval: float = 1.0,
        disable_system_prompt: bool = False,
        aws_access_key: str | None = None,
        aws_secret_key: str | None = None,
        aws_region: str | None = None,
    ):
        self.model_name = model_name
        self.max_retries = max_retries
        self.async_request_interval = async_request_interval
        self.disable_system_prompt = disable_system_prompt

        self.anthropic_client = AnthropicBedrockClient(
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_region=aws_region,
        )

    async def async_request(
        self,
        prompt: list[str],
        response: list[str | None],
        system_prompt: str | None = None,
        sampling_params: MutableMapping | None = None,
    ) -> str:
        if sampling_params is None:
            sampling_params = {}

        messages = await asyncio.to_thread(self.get_messages, prompt, response)

        sampling_params = dict(sampling_params)
        # Ignore unsupported parameters
        for key in ["seed", "frequency_penalty"]:
            if key in sampling_params:
                warnings.warn(f"BedrockAnthropic does not support {key} parameter. Ignoring.")
                sampling_params.pop(key)

        completions: Message
        if system_prompt is not None:
            completions = self.anthropic_client.messages.create(
                model=self.model_name,
                messages=cast(list[MessageParam], messages),
                system=system_prompt,
                **sampling_params,
            )
        else:
            completions = self.anthropic_client.messages.create(
                model=self.model_name,
                messages=cast(list[MessageParam], messages),
                **sampling_params,
            )

        assert isinstance(completions.content[0], TextBlock)
        return completions.content[0].text
