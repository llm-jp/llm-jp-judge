# llm-jp-gen-eval(仮)

生成自動評価を行うパッケージです。  
開発中のアルファ版です。  

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requrements.txt
```

## Setup

生成もしくは評価に使用するAPIの情報を`.env`ファイルに入力して下さい。  
詳しくは[Client](#client)を参照ください。

## How to Use

llm-jp-gen-evalでは生成と評価を分けて行います。  

### Generate

```bash
MODEL_NAME=llm-jp/llm-jp-3-1.8b-instruct
CACHE_DIR=./output/llm-jp-3-1.8b-instruct
python3 -m src.llm_jp_gen_eval.generate \
    output.dir=$CACHE_DIR \
    output.overwrite=true \
    client=vllm \
    client.model_name=$MODEL_NAME \
    benchmark.ichikara.dataset.path=/Path/to/ichikara-eval-test.json \
    benchmark.answer_carefully.dataset.path=/Path/to/AnswerCarefullyVersion001_Test.json
```

### Evaluate

```bash
python3 -m src.llm_jp_gen_eval.evaluate \
    input.dir=$CACHE_DIR \
    client=azure \
    client.model_name=gpt-4o-2024-08-06 \
    client.async_request_interval=0.5 \
    dashboard=wandb \
    dashboard.entity={entity_name} \
    dashboard.project={project_name} \
    dashboard.run_name={run_name}
```

## 仕様

### Benchmark

#### Ichikara-eval

**TBA**

#### Answer Carefully

**TBA**

#### MT-Bench

**TBA***

### Client

生成もしくは評価に使用可能な推論用クライアントは以下の通りです。

#### Azure(`client=azure`)

Azure OpenAI APIのモデル(例:`gpt-4o-2024-08-06`)を指定できます。  
`.env`ファイルを作成し、以下の環境変数を追記してください。
```bash:.env
AZURE_ENDPOINT="https://********.openai.azure.com/"
AZURE_API_KEY="********"
```

#### Bedrock(`client=bedrock`)

AWS Bedrock APIのモデル(例:`anthropic.claude-3-5-sonnet-20240620-v1:0`)を指定できます。  
`.env`ファイルを作成し、以下の環境変数を追記してください。
```bash:.env
AWS_ACCESS_KEY="********"
AWS_SECRET_KEY="****************"
AWS_REGION="us-west-2"
```


#### vLLM(`client=vllm`)

- `vllm` : Hugging Faceのモデル名もしくはパス(例:`llm-jp/llm-jp-3-1.8b-instruct`)を指定できます。vLLMを使用してローカルで推論を行います。チャットテンプレートに対応したモデルを指定する必要があります。
