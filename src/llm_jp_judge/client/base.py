from copy import deepcopy


class BaseClient:
    def get_messages(self, prompt, response, system_prompt=None):
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

    def fill_sampling_params(self, sampling_params):
        return {k: v for k, v in sampling_params.items() if v is not None}
