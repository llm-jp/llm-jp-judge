from llm_jp_judge.client.base import BaseClient
from llm_jp_judge.client.remote import AzureOpenAI, BedrockAnthropic, OpenAI


def load_client(name: str = "azure", **kwargs) -> BaseClient:
    if name == "openai":
        return OpenAI(**kwargs)
    elif name == "azure":
        return AzureOpenAI(**kwargs)
    elif name == "bedrock":
        return BedrockAnthropic(**kwargs)
    raise ValueError(f"Invalid client name: {name}")
