import re

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

    def get_messages(self, prompt, system_prompt=None):
        if self.disable_system_prompt and system_prompt is not None:
            prompt = f"{system_prompt}\n\n{prompt}"
            system_prompt = None

        if system_prompt is None:
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

        return messages

    def batch_request(
        self,
        prompts,
        system_prompt=None,
        sampling_params={},
    ):
        sampling_params = SamplingParams(**sampling_params)

        messages_list = []
        for prompt in prompts:
            messages = self.get_messages(prompt, system_prompt=system_prompt)
            messages_list.append(messages)

        outputs = self.llm.chat(messages_list, sampling_params=sampling_params)
        responses = [output.outputs[0].text for output in outputs]
        return responses

    def __call__(self, data, regex=None, system_prompt=None, sampling_params={}):
        prompts = [d["prompt"] for d in data]
        pending_indices = list(range(len(prompts)))

        for d in data:
            d["response"], d["pattern"], d["error_messages"] = None, None, []

        retry_count = 0
        done_indices = set()
        while retry_count < self.max_retries and len(pending_indices) > 0:
            responses = self.batch_request(
                [prompts[i] for i in pending_indices],
                system_prompt=system_prompt,
                sampling_params=sampling_params,
            )

            for idx, response in zip(pending_indices, responses):
                data[idx]["response"] = response
                if regex is None:
                    done_indices.add(idx)
                    continue

                try:
                    m = re.search(regex, response)
                    data[idx]["pattern"] = m.group(1)
                except Exception as e:
                    data[idx]["error_messages"].append(str(e))
                    continue

                done_indices.add(idx)
                continue

            pending_indices = list(set(pending_indices) - done_indices)
            retry_count += 1

        return data
