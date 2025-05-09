import os
import logging
import warnings
import asyncio

import openai
from openai import OpenAI as OpenAIClient
from openai import AzureOpenAI as AzureOpenAIClient
from anthropic import AnthropicBedrock as AnthropicBedrockClient

from dotenv import load_dotenv

import tqdm
import tqdm.asyncio

from .local import BaseClient

load_dotenv(override=True)


class OpenAI(BaseClient):
    def __init__(
        self,
        model_name="gpt-4o-2024-08-06",
        max_retries=1,
        async_request_interval=1.0,
        disable_system_prompt=False,
    ):
        self.model_name = model_name
        self.max_retries = max_retries
        self.async_request_interval = async_request_interval
        self.disable_system_prompt = disable_system_prompt

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            logging.warning("Environment variable OPENAI_API_KEY is not set.")
            api_key = input("Enter Azure OpenAI API key: ")

        base_url = os.getenv("OPENAI_BASE_URL", None)

        organization = os.getenv("OPENAI_ORGANIZATION", None)

        project = os.getenv("OPENAI_PROJECT", None)

        self.client = OpenAIClient(
            api_key=api_key,
            organization=organization,
            project=project,
            base_url=base_url,
        )

    async def async_request(
        self,
        prompt,
        response,
        system_prompt=None,
        sampling_params={},
    ):
        messages = await asyncio.to_thread(
            self.get_messages, prompt, response, system_prompt=system_prompt
        )

        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model_name,
            messages=messages,
            **sampling_params,
        )
        return response.choices[0].message.content

    async def process_data(
        self, data, score_extractor=None, system_prompt=None, sampling_params={}
    ):
        tasks = []
        wait = 0
        for d in data:
            if type(d["prompt"]) == str:  # Single turn
                d["is_single_turn"] = True
                d["prompt"] = [d["prompt"]]
            elif type(d["prompt"]) == list:  # Multi turn
                d["is_single_turn"] = False

            tasks.append(
                self._process_single_request(
                    d,
                    score_extractor,
                    system_prompt,
                    wait=wait,
                    sampling_params=sampling_params,
                )
            )
            wait += self.async_request_interval * len(d["prompt"])

        data = await tqdm.asyncio.tqdm.gather(*tasks, desc=self.model_name)

        for d in data:
            is_single_turn = d.pop("is_single_turn")
            if is_single_turn:
                d["prompt"] = d["prompt"][0]
                d["response"] = d["response"][0]
                d["pattern"] = d["pattern"][0]
                d["error_messages"] = d["error_messages"][0]

        return data

    async def _process_single_request(
        self, d, score_extractor, system_prompt, wait, sampling_params={}
    ):
        await asyncio.sleep(wait)

        d["response"], d["pattern"], d["error_messages"] = [], [], []
        for turn in range(len(d["prompt"])):
            retry_count = 0
            sleep = 0

            d["response"].append(None)
            d["pattern"].append(None)
            d["error_messages"].append([])
            while retry_count < self.max_retries:
                if len(d["error_messages"][-1]) > 0:
                    logging.warning(
                        f"{d['error_messages'][-1][-1]}. Retrying in {sleep} seconds."
                    )
                await asyncio.sleep(sleep)

                try:
                    d["response"][-1] = await self.async_request(
                        d["prompt"][: turn + 1],
                        d["response"][:turn],
                        system_prompt=system_prompt,
                        sampling_params=sampling_params,
                    )
                except (openai.RateLimitError, openai.APITimeoutError) as e:
                    d["error_messages"][-1].append(str(e))
                    sleep = 60
                except openai.BadRequestError as e:
                    d["error_messages"][-1].append(str(e))

                    retry_count += 1
                    sleep = self.async_request_interval
                    await asyncio.sleep(self.async_request_interval)
                else:
                    if score_extractor is not None:
                        try:
                            d["pattern"][-1] = score_extractor(d["response"][-1])
                        except Exception as e:
                            d["error_messages"][-1].append(str(e))
                            retry_count += 1
                            sleep = self.async_request_interval
                            continue
                    break

            if turn < len(d["prompt"]) - 1:
                await asyncio.sleep(self.async_request_interval)

        return d

    def __call__(
        self, data, score_extractor=None, system_prompt=None, sampling_params={}
    ):
        sampling_params = self.fill_sampling_params(sampling_params)

        return asyncio.run(
            self.process_data(
                data, score_extractor, system_prompt, sampling_params=sampling_params
            )
        )


class AzureOpenAI(OpenAI):

    def __init__(
        self,
        model_name="gpt-4o-2024-08-06",
        max_retries=1,
        async_request_interval=1.0,
        disable_system_prompt=False,
    ):
        self.model_name = model_name
        self.max_retries = max_retries
        self.async_request_interval = async_request_interval
        self.disable_system_prompt = disable_system_prompt

        api_key = os.getenv("AZURE_API_KEY")
        if api_key is None:
            logging.warning("Environment variable AZURE_API_KEY is not set.")
            api_key = input("Enter Azure OpenAI API key: ")

        api_endpoint = os.getenv("AZURE_ENDPOINT")
        if api_endpoint is None:
            logging.warning("Environment variable AZURE_ENDPOINT is not set.")
            api_endpoint = input("Enter Azure OpenAI API endpoint: ")

        api_version = os.getenv("AZURE_API_VERSION", "2023-05-15")

        self.client = AzureOpenAIClient(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=api_endpoint,
        )


class BedrockAnthropic(AzureOpenAI):
    def __init__(
        self,
        model_name="anthropic.claude-3-5-sonnet-20240620-v1:0",
        max_retries=1,
        async_request_interval=1.0,
        disable_system_prompt=False,
    ):
        self.model_name = model_name
        self.max_retries = max_retries
        self.async_request_interval = async_request_interval
        self.disable_system_prompt = disable_system_prompt

        aws_access_key = os.getenv("AWS_ACCESS_KEY")
        if aws_access_key is None:
            logging.warning("Environment variable AWS_ACCESS_KEY is not set.")
            aws_access_key = input("Enter AWS Bedrock access key: ")

        aws_secret_key = os.getenv("AWS_SECRET_KEY")
        if aws_secret_key is None:
            logging.warning("Environment variable AWS_SECRET_KEY is not set.")
            aws_secret_key = input("Enter AWS Bedrock secret key: ")

        aws_region = os.getenv("AWS_REGION")
        if aws_region is None:
            logging.warning("Environment variable AWS_REGION is not set.")
            aws_region = input("Enter AWS Bedrock region: ")

        self.client = AnthropicBedrockClient(
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_region=aws_region,
        )

    async def async_request(
        self,
        prompt,
        response,
        system_prompt=None,
        sampling_params={},
    ):
        messages = await asyncio.to_thread(self.get_messages, prompt, response)

        sampling_params = dict(sampling_params)
        # Ignore unsupported parameters
        for key in ["seed", "frequency_penalty"]:
            if key in sampling_params:
                warnings.warn(
                    f"BedrockAnthropic does not support {key} parameter. Ignoring."
                )
                sampling_params.pop(key)

        completions = self.client.messages.create(
            model=self.model_name,
            messages=messages,
            system=system_prompt,
            **sampling_params,
        )

        return completions.content[0].text
