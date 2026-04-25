import pandas as pd
import re
import os
import json
from config_reconstruct import ReconstructConfig

cfg = ReconstructConfig()

os.makedirs(cfg.output_dir, exist_ok=True)

df_train = pd.read_csv(cfg.train_path)
df_test = pd.read_csv(cfg.test_path)

df = pd.concat([df_train, df_test]).drop_duplicates(subset=[cfg.word_id_column], keep='first')

regex = r'^(.*)_page_(\d+)_(\d+)$'
parsed = df[cfg.word_id_column].str.extract(regex)
df['text_id'] = parsed[0]
df['page_num'] = parsed[1].astype(int)
df['word_idx'] = parsed[2].astype(int)

word_id_map = df[[cfg.word_id_column, 'text_id', 'page_num', 'word_idx']]
word_id_map.to_csv(os.path.join(cfg.output_dir, cfg.word_id_map_filename), index=False)

grouped = df.sort_values(['text_id', 'page_num', 'word_idx']).groupby(['text_id', 'page_num'])

reconstructed_data = []

for (text_id, page_num), group in grouped:
    reconstructed_text = " ".join(group[cfg.word_column].astype(str).tolist())
    word_id_sequence = json.dumps(group[cfg.word_id_column].tolist())
    word_sequence = json.dumps(group[cfg.word_column].tolist())
    
    reconstructed_data.append({
        'text_id': text_id,
        'page_num': page_num,
        'reconstructed_text': reconstructed_text,
        'word_id_sequence': word_id_sequence,
        'word_sequence': word_sequence
    })

reconstructed_df = pd.DataFrame(reconstructed_data)
reconstructed_df.to_csv(os.path.join(cfg.output_dir, cfg.reconstructed_pages_filename), index=False)

page_counts = grouped.size()
print(f"{len(reconstructed_df)}")
print(f"{len(df)}")
print(f"{page_counts.mean()}")
print(f"{page_counts.min()}")
print(f"{page_counts.max()}")
