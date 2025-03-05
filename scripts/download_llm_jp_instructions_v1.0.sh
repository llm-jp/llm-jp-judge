python -c "import os, datasets, pandas as pd; \
output_dir = './data/cache/llm-jp/llm-jp-instructions/v1.0'; \
ds = datasets.load_dataset('llm-jp/llm-jp-instructions', 'v1.0'); \
os.makedirs(output_dir, exist_ok=True); \
[ds[split].to_pandas().to_json(os.path.join(output_dir, f'{split}.json'), orient='records', indent=2, force_ascii=False) for split in ds]; \
print(f'Successfully downloaded the llm-jp-instructions dataset to {output_dir}')"