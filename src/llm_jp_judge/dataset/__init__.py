from pydantic import BaseModel


class DatasetItem(BaseModel):
    """Base class for dataset item.

    Attributes:
        ID: Unique identifier of the item.
        prompt: Prompt for each turn.
        response: Model response for each turn.
        error_messages: Error messages for each turn.
        pattern: Extracted pattern for each turn.
        original_index: Original index of the item.
    """

    ID: int | str
    prompt: list[str]
    response: list[str | None] = []
    error_messages: list[list[str]] = []
    pattern: list[str | dict[str, int] | None] = []
    original_index: int | None = None


class DatasetItemForEvaluation(DatasetItem):
    """Base class for dataset item for evaluation.

    Attributes:
        generate_prompt: Prompt for each turn used in generation phase.
        generate_response: Model response for each turn used in generation phase.
        generate_errors: Error messages for each turn used in generation phase.
        metric: Metric used for evaluation.
    """

    generate_prompt: list[str] = []
    generate_response: list[str | None] = []
    generate_errors: list[list[str]] = []
    metric: str | None = None
