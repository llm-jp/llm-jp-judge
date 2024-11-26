import re
import time
from openai import AzureOpenAI as AzureOpenAIClient
from anthropic import AnthropicBedrock as AnthropicBedrockClient


class AzureOpenAI:
    def __init__(
        self,
        model_name="gpt-4o-2024-05-13",
        api_key=None,
        api_version="2023-05-15",
        api_endpoint=None,
        max_tokens=128,
        max_retries=1,
        request_interval=1.0,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.request_interval = request_interval

        self.client = AzureOpenAIClient(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=api_endpoint,
        )

    def request(self, prompt, system_prompt=None):
        if system_prompt is None:
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    def __call__(self, data, regex=None, system_prompt=None):
        for d in data:
            retry_count = 0
            d["response"], d["pattern"], d["error_messages"] = None, None, []
            while retry_count < self.max_retries:
                d["response"] = self.request(d["prompt"], system_prompt=system_prompt)
                time.sleep(self.request_interval)

                if regex is None:
                    break

                try:
                    m = re.search(regex, d["response"])
                    d["pattern"] = m.group(1)
                except Exception as e:
                    d["error_messages"].append(str(e))
                    retry_count += 1
                    continue

                break

        return data


class BedrockAnthropic(AzureOpenAI):
    def __init__(
        self,
        model_name="anthropic.claude-3-5-sonnet-20240620-v1:0",
        aws_access_key=None,
        aws_secret_key=None,
        aws_region="us-east-1",
        max_tokens=128,
        max_retries=1,
        request_interval=1.0,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.request_interval = request_interval

        self.client = AnthropicBedrockClient(
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_region=aws_region,
        )

    def request(self, prompt, system_prompt=None):
        messages = [{"role": "user", "content": prompt}]

        completions = self.client.messages.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            system=system_prompt,
        )

        return completions.content[0].text
