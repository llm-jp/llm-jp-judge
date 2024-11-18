## Install

```
python3 -m venv venv
source venv/bin/activate
pip install -r requrements.txt
```

## Quality Evaluation

```
INFERENCE_MODEL_NAME="llm-jp/llm-jp-3-1.8b-instruct"
EVALUATE_MODEL_NAME="gpt-4o-2024-05-13"

INPUT_FILE="./example/data/quality.jsonl"
RESPONSE_FILE="./example/output/${INFERENCE_MODEL_NAME}/quality.jsonl"

python3 -m src.llm_jp_gen_eval.inference \
    input.path=$INPUT_FILE \
    output.path=$RESPONSE_FILE \
    output.overwrite=true \
    client=vllm \
    client.model_name=$INFERENCE_MODEL_NAME

python3 -m src.llm_jp_gen_eval.evaluate \
    input.path=$RESPONSE_FILE \
    aspect=quality \
    client=azure \
    client.model_name=$EVALUATE_MODEL_NAME \
    client.api_key=****** \
    client.api_endpoint=https://******.openai.azure.com/  \
    client.max_retries=1
```

## Safety Evaluation

```
INFERENCE_MODEL_NAME="llm-jp/llm-jp-3-1.8b-instruct"
EVALUATE_MODEL_NAME="gpt-4o-2024-05-13"

INPUT_FILE="./example/data/safety.jsonl"
RESPONSE_FILE="./example/output/${INFERENCE_MODEL_NAME}/safety.jsonl"

python3 -m src.llm_jp_gen_eval.inference \
    input.path=$INPUT_FILE \
    output.path=$RESPONSE_FILE \
    output.overwrite=true \
    client=vllm \
    client.model_name=$INFERENCE_MODEL_NAME

python3 -m src.llm_jp_gen_eval.evaluate \
    input.path=$RESPONSE_FILE \
    aspect=safety \
    client=azure \
    client.model_name=$EVALUATE_MODEL_NAME \
    client.api_key=****** \
    client.api_endpoint=https://******.openai.azure.com/  \
    client.max_retries=1
```