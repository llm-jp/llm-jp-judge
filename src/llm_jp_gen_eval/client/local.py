import re
from copy import deepcopy

import hydra

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

NUM_GPUS = torch.cuda.device_count()


class vLLMClient:
    def __init__(
        self,
        model_name="llm-jp/llm-jp-3-13b-instruct",
        batch_size=1,
        download_dir="~/.cache/huggingface",
        max_retries=1,
        disable_system_prompt=False,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.disable_system_prompt = disable_system_prompt

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.llm = LLM(
            model=self.model_name,
            download_dir=hydra.utils.to_absolute_path(download_dir),
            tensor_parallel_size=NUM_GPUS,
        )

    def get_messages(self, prompt, response, system_prompt=None):
        if self.disable_system_prompt and system_prompt is not None:
            prompt = deepcopy(prompt)
            prompt[0] = f"{system_prompt}\n\n{prompt[0]}"
            system_prompt = None

        messages = []
        for i in range(len(prompt)):
            messages.append({"role": "user", "content": prompt[i]})
            if i > 0:
                messages.append({"role": "assistant", "content": response[i - 1]})

        if system_prompt is not None:
            messages.insert(0, {"role": "system", "content": system_prompt})

        return messages

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

        outputs = self.llm.chat(messages_list, sampling_params=sampling_params)
        responses = [output.outputs[0].text for output in outputs]
        return responses

    def _process_turn_requests(
        self, data, turn, regex=None, system_prompt=None, sampling_params={}
    ):
        pending_indices = [i for i, d in enumerate(data) if len(d["prompt"]) > turn]

        tmp_data = [
            {"response": None, "pattern": None, "error_messages": []} for _ in data
        ]

        retry_count = 0
        done_indices = set()
        while retry_count < self.max_retries and len(pending_indices) > 0:
            responses = self.batch_request(
                [data[i]["prompt"][: turn + 1] for i in pending_indices],
                [data[i]["response"] for i in pending_indices],
                system_prompt=system_prompt,
                sampling_params=sampling_params,
            )

            for idx, response in zip(pending_indices, responses):
                tmp_data[idx]["response"] = response
                if regex is not None:
                    try:
                        m = re.search(regex, response)
                        tmp_data[idx]["pattern"] = m.group(1)
                    except Exception as e:
                        tmp_data[idx]["error_messages"].append(str(e))
                        continue

                data[idx]["response"].append(tmp_data[idx]["response"])
                data[idx]["pattern"].append(tmp_data[idx]["pattern"])
                data[idx]["error_messages"].append(tmp_data[idx]["error_messages"])

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
        return self.process_data(
            data, regex, system_prompt, sampling_params=sampling_params
        )
