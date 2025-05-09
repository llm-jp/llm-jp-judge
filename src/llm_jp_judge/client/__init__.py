from .remote import OpenAI, AzureOpenAI, BedrockAnthropic
from .local import vLLMClient


def load_client(name="azure", **kwargs):
    if name == "openai":
        return OpenAI(**kwargs)
    elif name == "azure":
        return AzureOpenAI(**kwargs)
    elif name == "bedrock":
        return BedrockAnthropic(**kwargs)
    elif name == "vllm":
        return vLLMClient(**kwargs)
    raise ValueError(f"Invalid client name: {name}")
