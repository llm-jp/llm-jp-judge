uv run python -c "import os, datasets, pandas as pd; \
output_dir = './data/cache/ScaleAI/MultiChallenge/test'; \
ds = datasets.load_dataset('ScaleAI/MultiChallenge'); \
os.makedirs(output_dir, exist_ok=True); \
[ds[split].to_pandas().to_json(os.path.join(output_dir, f'{split}.json'), orient='records', indent=2, force_ascii=False) for split in ds]; \
print(f'Successfully downloaded the MultiChallenge dataset to {output_dir}')"
