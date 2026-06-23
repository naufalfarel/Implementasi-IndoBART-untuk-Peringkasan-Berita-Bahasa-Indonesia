import json

path = r'notebook/IndoBART_Summarization_UAS.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if 'df[\"article\"] = df[\"article\"].apply(clean_text)' in src:
            new_src = src.replace('df[\"article\"] = df[\"article\"].apply(clean_text)', 'tqdm.pandas(desc=f\"Cleaning {name} article\")\n    df[\"article\"] = df[\"article\"].progress_apply(clean_text)')
            new_src = new_src.replace('df[\"summary\"] = df[\"summary\"].apply(clean_text)', 'tqdm.pandas(desc=f\"Cleaning {name} summary\")\n    df[\"summary\"] = df[\"summary\"].progress_apply(clean_text)')
            cell['source'] = [line + '\n' for line in new_src.split('\n')]
            if cell['source'][-1] == '\n':
                cell['source'] = cell['source'][:-1]
            else:
                cell['source'][-1] = cell['source'][-1].rstrip('\n')

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Updated cell 11 to use progress_apply.')