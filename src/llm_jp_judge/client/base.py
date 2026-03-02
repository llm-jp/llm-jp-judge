from copy import deepcopy

from omegaconf import DictConfig


class BaseClient:
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

        messages = []
        for turn in range(len(prompt)):
            messages.append({"role": "user", "content": prompt[turn]})
            if turn < len(response):
                messages.append({"role": "assistant", "content": response[turn]})

        if system_prompt is not None:
            messages.insert(0, {"role": "system", "content": system_prompt})

        return messages

    def fill_sampling_params(
        self, sampling_params: dict[str, int | float | None] | DictConfig
    ) -> dict[str, int | float]:
        return {k: v for k, v in sampling_params.items() if v is not None}
