from pydantic import BaseModel


class DatasetItem(BaseModel):
    ID: int | str
    prompt: list[str]
    response: list[str | None] = []
    error_messages: list[list[str]] = []
    pattern: list[str | dict[str, int] | None] = []
    original_index: int | None = None


class DatasetItemForEvaluation(DatasetItem):
    generate_prompt: list[str] = []
    generate_response: list[str | None] = []
    generate_errors: list[list[str]] = []
    metric: str | None = None
