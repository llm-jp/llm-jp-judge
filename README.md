# llm-jp-judge

生成自動評価を行うパッケージです。  
開発中のアルファ版です。  

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requrements.txt
```

## Download Dataset

以下のデータセットをダウンロードします。  
(既にローカルに保存されたデータを用いる場合は必要ありません。)

- [AnswerCarefully v2.0](https://huggingface.co/datasets/llm-jp/AnswerCarefully)
  1. huggingface-cliへのログイン
    ```bash
    huggingface-cli login
    ```
  2. データセットへの[アクセス申請](https://huggingface.co/datasets/llm-jp/AnswerCarefully)
  3. ダウンロード
    ```bash
    bash scripts/download_ac_v2.0.sh
    ```

## Setup

生成もしくは評価に使用するAPIの情報を`.env`ファイルに入力して下さい。  
詳しくは[Client](#client)を参照ください。

## How to Use

llm-jp-gen-evalでは生成と評価を分けて行います。  
以下は、`llm-jp/llm-jp-3-1.8b-instruct`により生成を行い、gpt-4oにより評価する例です。  

### Generation

```bash
MODEL_NAME=llm-jp/llm-jp-3-1.8b-instruct
CACHE_DIR=./output/llm-jp-3-1.8b-instruct
python3 -m src.llm_jp_gen_eval.generate \
    output.dir=$CACHE_DIR \
    output.overwrite=true \
    client=vllm \
    client.model_name=$MODEL_NAME \
    benchmark.ichikara.dataset.path=/Path/to/ichikara-eval-test.json \
    benchmark.answer_carefully.dataset.path=./data/cache/llm-jp/AnswerCarefully/v2.0/test.json
```

### Evaluation

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

## Benchmark

### Ichikara-eval

**TBA**

### Answer Carefully

**TBA**

### MT-Bench: `benchmark.mt_bench`

参照: [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)

データセットパスの指定は必要ありません。  
非アクティブにしたい場合は、生成時に`benchmark.mt_bench.dataset.path=null`を指定してください。

### Japanese MT-Bench: `benchmark.ja_mt_bench`

参照: [Japanese MT-Bench](https://github.com/Stability-AI/FastChat)

データセットパスの指定は必要ありません。  
非アクティブにしたい場合は、生成時に`benchmark.ja_mt_bench.dataset.path=null`を指定してください。

## Client

生成もしくは評価に使用可能な推論用クライアントは以下の通りです。

### Azure

Azure OpenAI APIのデプロイ名(例:`gpt-4o-2024-08-06`)を指定できます。

```
python3 -m src.llm_jp_gen_eval.evaluate \ # generate or evaluate
    client=azure \
    client.model_name=gpt-4o-2024-08-06 \  # デプロイ名
    client.async_request_interval=0.5  # APIリクエストの間隔(秒)
```

実行ディレクトリに`.env`ファイルを作成し、以下の環境変数を追記してください。
```bash:.env
AZURE_ENDPOINT="https://********.openai.azure.com/"
AZURE_API_KEY="********"
```

### Bedrock

AWS Bedrock APIのデプロイ名(例:`anthropic.claude-3-5-sonnet-20240620-v1:0`)を指定できます。  

```
python3 -m src.llm_jp_gen_eval.evaluate \ # generate or evaluate
    client=bedrock \
    client.model_name=anthropic.claude-3-5-sonnet-20240620-v1:0 \  # デプロイ名
    client.async_request_interval=10  # APIリクエストの間隔(秒)
```

実行ディレクトリに`.env`ファイルを作成し、以下の環境変数を追記してください。
```bash:.env
AWS_ACCESS_KEY="********"
AWS_SECRET_KEY="****************"
AWS_REGION="us-west-2"
```

### vLLM: `client=vllm`

Hugging Faceのモデル名(例:`llm-jp/llm-jp-3-1.8b-instruct`)もしくはパスを指定できます。  
vLLMを使用してローカルで推論を行います。
> [!NOTE]
> モデルが使用するトークナイザーがチャットテンプレートに対応している必要があります。  
> 対応していない場合、チャットテンプレートに対応したトークナイザーを`client.tokenizer_name`として指定するか、jinja形式のチャットテンプレートを`client.chat_template.path`として与えてください。

```bash
python3 -m src.llm_jp_gen_eval.evaluate \ # generate or evaluate
    client=vllm \
    client.model_name=$INFERENCE_MODEL_NAME # Huggin Faceのモデル名 or パス
```

