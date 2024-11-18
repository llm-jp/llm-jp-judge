from .remote import AzureOpenAI
from .local import vLLMClient


def load_client(name="azure", **kwargs):
    if name == "azure":
        return AzureOpenAI(**kwargs)
    elif name == "vllm":
        return vLLMClient(**kwargs)
    raise ValueError(f"Invalid client name: {name}")
