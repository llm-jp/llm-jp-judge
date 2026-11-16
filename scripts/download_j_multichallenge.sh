uv run python -c "import os, datasets, pandas as pd; \
output_dir = './data/cache/sbintuitions/J-MultiChallenge/test'; \
ds = datasets.load_dataset('sbintuitions/J-MultiChallenge'); \
os.makedirs(output_dir, exist_ok=True); \
[ds[split].to_pandas().to_json(os.path.join(output_dir, f'{split}.json'), orient='records', indent=2, force_ascii=False) for split in ds]; \
print(f'Successfully downloaded the J-MultiChallenge dataset to {output_dir}')"
