import re

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


class vLLMClient:
    def __init__(
        self,
        model_name="llm-jp/llm-jp-3-13b-instruct",
        batch_size=1,
        max_tokens=128,
        download_dir="~/.cache/huggingface",
        max_retries=1,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.llm = LLM(model=self.model_name, download_dir=download_dir)
        self.sampling_params = SamplingParams(max_tokens=self.max_tokens)

    def batch_request(self, prompts, system_prompt=None):
        messages_list = []
        for prompt in prompts:
            if system_prompt is None:
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
            messages_list.append(messages)

        outputs = self.llm.chat(messages_list, sampling_params=self.sampling_params)
        responses = [output.outputs[0].text for output in outputs]
        return responses

    def __call__(self, data, regex=None, system_prompt=None):
        prompts = [d["prompt"] for d in data]
        pending_indices = list(range(len(prompts)))

        for d in data:
            d["response"], d["pattern"], d["error_messages"] = None, None, []

        retry_count = 0
        done_indices = set()
        while retry_count < self.max_retries and len(pending_indices) > 0:
            responses = self.batch_request(
                [prompts[i] for i in pending_indices], system_prompt=system_prompt
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
