import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from config_morph import MorphConfig
cfg = MorphConfig()
assert str(cfg.gpu_id) == os.environ["CUDA_VISIBLE_DEVICES"], \
    f"Config gpu_id {cfg.gpu_id} does not match CUDA_VISIBLE_DEVICES {os.environ['CUDA_VISIBLE_DEVICES']}"

import pandas as pd
import spacy
import json
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

spacy.prefer_gpu()
nlp = spacy.load(cfg.spacy_model)

df = pd.read_csv(os.path.join(cfg.output_dir, cfg.reconstructed_pages_filename))
df[cfg.word_id_sequence_column] = df[cfg.word_id_sequence_column].apply(json.loads)
df[cfg.word_sequence_column] = df[cfg.word_sequence_column].apply(json.loads)

results = []
total_pages = len(df)
iterator = tqdm(range(total_pages)) if tqdm else range(total_pages)

for i in iterator:
    row = df.iloc[i]
    word_id_list = row[cfg.word_id_sequence_column]
    word_list = row[cfg.word_sequence_column]
    text = row[cfg.reconstructed_text_column]
    
    doc = nlp(text)
    
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
            morph = token.morph
            morph_count = len(morph)
            
            case_list = morph.get("Case")
            if not case_list:
                case_encoded = -1
            else:
                encoded_cases = [cfg.case_order.index(c) if c in cfg.case_order else -1 for c in case_list]
                case_encoded = max(encoded_cases)
                
            vf_list = morph.get("VerbForm")
            if not vf_list:
                verb_form = -1
            else:
                encoded_vf = [cfg.verb_form_order.index(v) if v in cfg.verb_form_order else -1 for v in vf_list]
                verb_form = max(encoded_vf)
                
            is_inflected = 1 if token.text.lower().strip() != token.lemma_.lower().strip() else 0
            
            results.append({
                'word_id': w_id,
                'morph_count': morph_count,
                'case_encoded': case_encoded,
                'verb_form': verb_form,
                'is_inflected': is_inflected
            })
        else:
            results.append({
                'word_id': w_id,
                'morph_count': 0,
                'case_encoded': -1,
                'verb_form': -1,
                'is_inflected': 0
            })
            
    if not tqdm and (i + 1) % 50 == 0:
        print(f"Processed {i + 1}/{total_pages} pages")

output_df = pd.DataFrame(results)
output_df.to_csv(os.path.join(cfg.output_dir, cfg.morph_output_filename), index=False)

word_id_map_df = pd.read_csv(os.path.join(cfg.output_dir, "word_id_map.csv"))
if len(output_df) != len(word_id_map_df):
    print(f"WARNING: word_id count mismatch. Expected {len(word_id_map_df)}, got {len(output_df)}")
if output_df['word_id'].duplicated().sum() > 0:
    print(f"WARNING: {output_df['word_id'].duplicated().sum()} duplicate word_ids found")
if output_df[['morph_count', 'case_encoded', 'verb_form', 'is_inflected']].isnull().sum().sum() > 0:
    print("WARNING: unexpected NaN values found")
if len(output_df) == len(word_id_map_df) and output_df['word_id'].duplicated().sum() == 0 and output_df[['morph_count', 'case_encoded', 'verb_form', 'is_inflected']].isnull().sum().sum() == 0:
    print("Validation passed.")

print("\n=== MORPH FEATURES SUMMARY ===")
print(f"Total words: {len(output_df)}")
mc_stats = output_df['morph_count']
print(f"morph_count — mean: {mc_stats.mean():.2f}, std: {mc_stats.std():.2f}, min: {mc_stats.min()}, max: {mc_stats.max()}")
print("case_encoded distribution:")
print(output_df['case_encoded'].value_counts())
print("verb_form distribution:")
print(output_df['verb_form'].value_counts())
inf_count = output_df['is_inflected'].sum()
print(f"is_inflected=1: {inf_count} ({(inf_count / len(output_df)) * 100:.2f}%)")
