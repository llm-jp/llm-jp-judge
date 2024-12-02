import os
import re
import logging
import asyncio

import openai
from openai import AzureOpenAI as AzureOpenAIClient
from anthropic import AnthropicBedrock as AnthropicBedrockClient

from dotenv import load_dotenv

import tqdm
import tqdm.asyncio

load_dotenv(override=True)


class AzureOpenAI:
    def __init__(
        self,
        model_name="gpt-4o-2024-05-13",
        max_tokens=128,
        max_retries=1,
        async_request_interval=1.0,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.async_request_interval = async_request_interval

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

    async def async_request(self, prompt, system_prompt=None):
        if system_prompt is None:
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    async def process_data(self, data, regex=None, system_prompt=None):
        tasks = []
        for i, d in enumerate(data):
            tasks.append(
                self._process_single_request(
                    d, regex, system_prompt, wait=i * self.async_request_interval
                )
            )
        return await tqdm.asyncio.tqdm.gather(*tasks, desc=self.model_name)

    async def _process_single_request(self, d, regex, system_prompt, wait):
        await asyncio.sleep(wait)

        retry_count = 0
        d["response"], d["pattern"], d["error_messages"] = None, None, []
        while retry_count < self.max_retries:
            try:
                d["response"] = await self.async_request(
                    d["prompt"], system_prompt=system_prompt
                )
            except openai.RateLimitError as e:
                d["error_messages"].append(str(e))
                retry_count += 1
                await asyncio.sleep(60)
            else:
                if regex is not None:
                    m = re.search(regex, d["response"])
                    try:
                        d["pattern"] = m.group(1)
                    except IndexError as e:
                        d["error_messages"].append(str(e))
                        retry_count += 1
                        continue
                break
        return d

    def __call__(self, data, regex=None, system_prompt=None):
        return asyncio.run(self.process_data(data, regex, system_prompt))


class BedrockAnthropic(AzureOpenAI):
    def __init__(
        self,
        model_name="anthropic.claude-3-5-sonnet-20240620-v1:0",
        max_tokens=128,
        max_retries=1,
        async_request_interval=1.0,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.async_request_interval = async_request_interval

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

    async def async_request(self, prompt, system_prompt=None):
        messages = [{"role": "user", "content": prompt}]

        completions = self.client.messages.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            system=system_prompt,
        )

        return completions.content[0].text
