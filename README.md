# llm-jp-gen-eval(仮)

生成自動評価を行うパッケージです。  
開発中のアルファ版です。  

## Installation

```
python3 -m venv venv
source venv/bin/activate
pip install -r requrements.txt
```

## Setup

生成もしくは評価に使用するAPIの情報を`.env`ファイルに入力して下さい。  
(vLLMを用いる場合は必要ありません。)

### Azure OpenAI API

```bash
AZURE_ENDPOINT="https://********.openai.azure.com/"
AZURE_API_KEY="********"
```

### AWS Bedrock API

```bash
AWS_ACCESS_KEY="********"
AWS_SECRET_KEY="****************"
AWS_REGION="us-west-2"
```

## How to Use

### Generate

評価用データセットに対する生成を行います。  
以下は、`llm-jp/llm-jp-3-1.8b-instruct`を対象にしています。

```
RESPONSE_DIR=./output/llm-jp/llm-jp-3-1.8b-instruct
python3 -m src.llm_jp_gen_eval.generate \
    output.dir=$RESPONSE_DIR \
    client=vllm \
    client.model_name=llm-jp/llm-jp-3-1.8b-instruct \
    benchmark.ichikara.dataset.path=./data/ichikara-instruction-eval-001-001-007.json \
    benchmark.answer_carefully.dataset.path=./data/AnswerCarefullyVersion001_Test.json
```

### Evaluation

生成されたデータに対する評価を行います。  
以下は、Azure OpenAI APIを用いて評価しています。  

評価結果のアップロードは`wandb`のみ対応しています。  

```
python3 -m src.llm_jp_gen_eval.evaluate \
    input.dir=$RESPONSE_DIR \
    client=azure \
    client.model_name=gpt-4o-2024-08-06 \
    dashboard=wandb \
    dashboard.entity=****** \
    dashboard.project=llm-jp-gen-eval
```

## 仕様

### 評価データセット

#### Ichikara-eval

TBA

#### Answer Carefully

TBA

### 推論用クライアント

生成もしくは評価に使用可能な推論用クライアントは以下の通りです。

- `azure` : Azure OpenAI APIのモデル(例:`gpt-4o-2024-08-06`)を指定できます。
- `bedrock` : AWS Bedrock APIのモデル(例:`anthropic.claude-3-5-sonnet-20240620-v1:0`)を指定できます。
- `vllm` : Hugging Faceのモデル名もしくはパス(例:`llm-jp/llm-jp-3-1.8b-instruct`)を指定できます。vLLMを使用してローカルで推論を行います。チャットテンプレートに対応したモデルを指定する必要があります。
