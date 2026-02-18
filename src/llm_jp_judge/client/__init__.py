from .remote import AzureOpenAI, BedrockAnthropic, OpenAI


def load_client(name="azure", **kwargs):
    if name == "openai":
        return OpenAI(**kwargs)
    elif name == "azure":
        return AzureOpenAI(**kwargs)
    elif name == "bedrock":
        return BedrockAnthropic(**kwargs)
    raise ValueError(f"Invalid client name: {name}")
