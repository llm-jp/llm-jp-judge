import re
from copy import deepcopy

import hydra

import torch
from vllm import LLM, SamplingParams
from huggingface_hub import repo_exists

NUM_GPUS = torch.cuda.device_count()


def load_chat_template(file_path):
    file_path = hydra.utils.to_absolute_path(file_path)
    with open(file_path, "r") as f:
        chat_template = f.read()
    return chat_template


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


class vLLMClient(BaseClient):
    def __init__(
        self,
        model_name="llm-jp/llm-jp-3-13b-instruct",
        tokenizer_name=None,
        batch_size=1,
        download_dir="~/.cache/huggingface",
        max_retries=1,
        chat_template={"path": None},
        disable_system_prompt=False,
    ):
        self.model_name = model_name
        if model_name.startswith((".", "/")) or not repo_exists(model_name):
            self.model_name = hydra.utils.to_absolute_path(model_name)

        self.tokenizer_name = tokenizer_name
        if tokenizer_name is not None and (
            tokenizer_name.startswith((".", "/")) or not repo_exists(tokenizer_name)
        ):
            self.tokenizer_name = hydra.utils.to_absolute_path(tokenizer_name)

        self.batch_size = batch_size
        self.max_retries = max_retries
        self.disable_system_prompt = disable_system_prompt

        self.chat_template = None
        if chat_template.get("path") is not None:
            self.chat_template = load_chat_template(chat_template["path"])

        download_dir = hydra.utils.to_absolute_path(download_dir)

        self.llm = LLM(
            model=self.model_name,
            tokenizer=self.tokenizer_name,
            download_dir=download_dir,
            tensor_parallel_size=NUM_GPUS,
        )

    def batch_request(
        self,
        prompts,
        responses,
        system_prompt=None,
        sampling_params={},
    ):
        sampling_params = SamplingParams(**sampling_params)

        messages_list = []
        for prompt, response in zip(prompts, responses):
            messages = self.get_messages(prompt, response, system_prompt=system_prompt)
            messages_list.append(messages)

        try:
            outputs = self.llm.chat(
                messages_list,
                sampling_params=sampling_params,
                chat_template=self.chat_template,
            )
        except ValueError:
            raise ValueError(
                f"No chat template found for {self.model_name}. Please provide a jinja style template with the argument client.chat_template.path=/path/to/chat_template.jinja."
            )

        responses = [output.outputs[0].text for output in outputs]
        return responses

    def _process_turn_requests(
        self, data, turn, regex=None, system_prompt=None, sampling_params={}
    ):
        pending_indices = [i for i, d in enumerate(data) if len(d["prompt"]) > turn]

        for d in data:
            d["response"].append(None)
            d["pattern"].append(None)
            d["error_messages"].append([])

        retry_count = 0
        done_indices = set()
        while retry_count < self.max_retries and len(pending_indices) > 0:
            responses = self.batch_request(
                [data[i]["prompt"][: turn + 1] for i in pending_indices],
                [data[i]["response"][:turn] for i in pending_indices],
                system_prompt=system_prompt,
                sampling_params=sampling_params,
            )

            for idx, response in zip(pending_indices, responses):
                assert response is not None, "Response is None"

                data[idx]["response"][-1] = response

                if regex is not None:
                    try:
                        m = re.search(regex, response)
                        data[idx]["pattern"][-1] = m.group(1)
                    except Exception as e:
                        data[idx]["error_messages"][-1].append(str(e))
                        continue

                done_indices.add(idx)
                continue

            pending_indices = list(set(pending_indices) - done_indices)
            retry_count += 1

        return data

    def process_data(self, data, regex=None, system_prompt=None, sampling_params={}):
        max_turn = 0
        for d in data:
            if type(d["prompt"]) == str:  # Single turn
                d["is_single_turn"] = True
                d["prompt"] = [d["prompt"]]
            elif type(d["prompt"]) == list:  # Multi turn
                d["is_single_turn"] = False
            max_turn = max(max_turn, len(d["prompt"]))
            d["response"], d["pattern"], d["error_messages"] = [], [], []

        for turn in range(max_turn):
            data = self._process_turn_requests(
                data, turn, regex, system_prompt, sampling_params
            )

        for d in data:
            is_single_turn = d.pop("is_single_turn")
            if is_single_turn:
                d["prompt"] = d["prompt"][0]
                d["response"] = d["response"][0]
                d["pattern"] = d["pattern"][0]
                d["error_messages"] = d["error_messages"][0]

        return data

    def __call__(self, data, regex=None, system_prompt=None, sampling_params={}):
        sampling_params = self.fill_sampling_params(sampling_params)

        return self.process_data(
            data, regex, system_prompt, sampling_params=sampling_params
        )
