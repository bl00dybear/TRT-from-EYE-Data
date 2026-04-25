import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from config_ner_pos import NerPosConfig
cfg = NerPosConfig()
assert str(cfg.gpu_id) == os.environ["CUDA_VISIBLE_DEVICES"], \
    f"Config gpu_id {cfg.gpu_id} does not match CUDA_VISIBLE_DEVICES {os.environ['CUDA_VISIBLE_DEVICES']}"

import pandas as pd
import spacy
import json
import subprocess

if not spacy.util.is_package(cfg.spacy_model):
    subprocess.run(["python", "-m", "spacy", "download", cfg.spacy_model], check=True)

spacy.prefer_gpu()
nlp = spacy.load(cfg.spacy_model)

df = pd.read_csv(os.path.join(cfg.output_dir, cfg.reconstructed_pages_filename))
df[cfg.word_id_sequence_column] = df[cfg.word_id_sequence_column].apply(json.loads)
df[cfg.word_sequence_column] = df[cfg.word_sequence_column].apply(json.loads)

results = []
total_pages = len(df)

for i, doc in enumerate(nlp.pipe(df[cfg.reconstructed_text_column], batch_size=cfg.batch_size)):
    row = df.iloc[i]
    word_id_list = row[cfg.word_id_sequence_column]
    word_list = row[cfg.word_sequence_column]
    
    word_spans = []
    char_cursor = 0
    for w_idx in range(len(word_list)):
        w_text = str(word_list[w_idx])
        word_spans.append((char_cursor, char_cursor + len(w_text), word_id_list[w_idx]))
        char_cursor += len(w_text) + 1
        
    word_to_tokens = {w_id: [] for w_id in word_id_list}
    for token in doc:
        if token.is_space:
            continue
        t_start = token.idx
        t_end = token.idx + len(token.text)
        for w_start, w_end, w_id in word_spans:
            if t_start < w_end and t_end > w_start:
                word_to_tokens[w_id].append(token)
                break
                
    for w_id in word_id_list:
        tokens = word_to_tokens[w_id]
        if tokens:
            token = tokens[0]
            is_NE = 1 if token.ent_type_ != "" else 0
            NE_type = token.ent_type_ if token.ent_type_ != "" else "O"
            POS_tag = token.pos_
            is_content_word = 1 if POS_tag in cfg.content_pos_tags else 0
            results.append({
                'word_id': w_id,
                'is_NE': is_NE,
                'NE_type': NE_type,
                'POS_tag': POS_tag,
                'is_content_word': is_content_word
            })
        else:
            results.append({
                'word_id': w_id,
                'is_NE': 0,
                'NE_type': "O",
                'POS_tag': "X",
                'is_content_word': 0
            })
            
    if (i + 1) % 50 == 0:
        print(f"Processed {i + 1}/{total_pages} pages")

output_df = pd.DataFrame(results)
output_df.to_csv(os.path.join(cfg.output_dir, cfg.ner_pos_output_filename), index=False)

print(f"{len(output_df)}")
print(f"{output_df['is_NE'].sum()} ({(output_df['is_NE'].sum() / len(output_df)) * 100:.2f}%)")
print(output_df['NE_type'].value_counts())
print(output_df['POS_tag'].value_counts())

word_id_map_df = pd.read_csv(os.path.join(cfg.output_dir, "word_id_map.csv"))
if len(output_df) != len(word_id_map_df):
    print(f"WARNING: word_id count mismatch. Expected {len(word_id_map_df)}, got {len(output_df)}")
if output_df['word_id'].duplicated().sum() > 0:
    print(f"WARNING: {output_df['word_id'].duplicated().sum()} duplicate word_ids found in ner_pos_features.csv")
null_cols = output_df[['is_NE', 'POS_tag', 'is_content_word']].columns[output_df[['is_NE', 'POS_tag', 'is_content_word']].isnull().any()].tolist()
if null_cols:
    print(f"WARNING: unexpected NaN values found in columns: {null_cols}")
if len(output_df) == len(word_id_map_df) and output_df['word_id'].duplicated().sum() == 0 and not null_cols:
    print("Validation passed.")
