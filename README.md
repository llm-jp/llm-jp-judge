<div align="center">

llm-jp-judge
===========================
<h4>日本語LLM-as-a-Judgeを統合的に扱えるようにするツール</h4>

<div align="left">  

# インストール

## 仮想環境

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requrements.txt
```

## データセット

以下のデータセットをダウンロードします。  
(既にローカルに保存されたデータを用いる場合は必要ありません。)

- [AnswerCarefully v2.0](https://huggingface.co/datasets/llm-jp/AnswerCarefully) (安全性評価用データセット)
  1. huggingface-cliへのログイン
      ```bash
      huggingface-cli login
      ```
  2. データセットへの[アクセス申請](https://huggingface.co/datasets/llm-jp/AnswerCarefully)
  3. ダウンロード
      ```bash
      bash scripts/download_ac_v2.0.sh
      ```

## 環境変数

必要に応じて生成もしくは評価に使用するAPIの情報を`.env`ファイルに入力して下さい。  

```bash:.env
# Microsoft Azure OpenAI Service
AZURE_ENDPOINT="https://********.openai.azure.com/"
AZURE_API_KEY="********"

# Amazon Bedrock API (Anthropic)
AWS_ACCESS_KEY="********"
AWS_SECRET_KEY="****************"
AWS_REGION="**-****-*" # e.g. us-west-2
```

# 使い方

llm-jp-gen-evalでは生成と評価を分けて行います。  
以下は、Hugging Face Hubの[llm-jp/llm-jp-3-1.8b-instruct](https://huggingface.co/llm-jp/llm-jp-3-1.8b-instruct)により生成を行い、gpt-4oにより評価する例です。  

## 生成

データセットに対して指定したモデルで生成を行います。  
各設定に関しては[ベンチマーク](#ベンチマーク)や[推論用クライアント](#推論用クライアント)を参照ください。

```bash
MODEL_NAME=llm-jp/llm-jp-3-1.8b-instruct
CACHE_DIR=./output/llm-jp-3-1.8b-instruct
python3 -m src.llm_jp_gen_eval.generate \
    output.dir=$CACHE_DIR \
    output.overwrite=true \
    client=vllm \
    client.model_name=$MODEL_NAME \
    benchmark.quality.dataset.path=/Path/to/ichikara-eval-test.json \
    benchmark.safety.dataset.path=./data/cache/llm-jp/AnswerCarefully/v2.0/test.json
```

## 評価 (LLM-as-a-Judge)

生成された結果に対して評価を行います。  

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

# ベンチマーク

## 品質評価 (日本語)

**TBA**

## 安全性評価 (日本語)

**TBA**

## MT-Bench (英語)
参照: [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)

データセットを内包しているため、データセットパスの指定は必要ありません。  
非アクティブにしたい場合は、生成時に以下のように指定してください。

```
python3 -m src.llm_jp_gen_eval.generate \
    benchmark.mt_bench.dataset.path=null
```

## MT-Bench (日本語)

参照: [Japanese MT-Bench](https://github.com/Stability-AI/FastChat)

データセットを内包しているため、データセットパスの指定は必要ありません。  
非アクティブにしたい場合は、生成時に以下のように指定してください。

```
python3 -m src.llm_jp_gen_eval.generate \
    benchmark.mt_bench.dataset.path=null
```

# 推論用クライアント

生成もしくは評価に使用可能な推論用クライアントは以下の通りです。

## Microsoft Azure OpenAI Service

Azure OpenAI APIのデプロイ名(例:`gpt-4o-2024-08-06`)を指定できます。

```
python3 -m src.llm_jp_gen_eval.evaluate \ # generate or evaluate
    client=azure \
    client.model_name=gpt-4o-2024-08-06 \  # デプロイ名
    client.async_request_interval=0.5  # APIリクエストの間隔(秒)
```

## Amazon Bedrock API (Anthropic)

AWS Bedrock APIのデプロイ名(例:`anthropic.claude-3-5-sonnet-20240620-v1:0`)を指定できます。  

```
python3 -m src.llm_jp_gen_eval.evaluate \ # generate or evaluate
    client=bedrock \
    client.model_name=anthropic.claude-3-5-sonnet-20240620-v1:0 \  # デプロイ名
    client.async_request_interval=10  # APIリクエストの間隔(秒)
```

## vLLM

vLLMを使用してローカルで推論を行います。  
Hugging Faceのモデル名(例:`llm-jp/llm-jp-3-1.8b-instruct`)もしくはパスを指定できます。  

> [!NOTE]
> モデルが使用するトークナイザーがチャットテンプレートに対応している必要があります。  
> 対応していない場合、チャットテンプレートに対応したトークナイザーを`client.tokenizer_name`として指定するか、jinja形式のチャットテンプレートを`client.chat_template.path`として与えてください。

```bash
python3 -m src.llm_jp_gen_eval.evaluate \ # generate or evaluate
    client=vllm \
    client.model_name=$INFERENCE_MODEL_NAME # Huggin Faceのモデル名 or パス
```

# ダッシュボード

評価結果を表示するためのダッシュボードを指定できます。  
現在はWandBのみサポートしています。

## WandB

```
python3 -m src.llm_jp_gen_eval.evaluate \
    dashboard=wandb \
    dashboard.entity={entity_name} \
    dashboard.project={project_name} \
    dashboard.run_name={run_name}
```
