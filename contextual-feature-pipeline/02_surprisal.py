import os
from config_surprisal import SurprisalConfig
cfg = SurprisalConfig()
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)

import pandas as pd
import torch
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
model.eval()
device = torch.device("cuda")
model.to(device)

df = pd.read_csv(os.path.join(cfg.output_dir, cfg.reconstructed_pages_filename))
df[cfg.word_id_sequence_column] = df[cfg.word_id_sequence_column].apply(json.loads)
df[cfg.word_sequence_column] = df[cfg.word_sequence_column].apply(json.loads)

results = []
nan_count = 0
total_pages = len(df)

iterator = tqdm(range(total_pages)) if tqdm else range(total_pages)

for i in iterator:
    row = df.iloc[i]
    text = row[cfg.reconstructed_text_column]
    word_ids = row[cfg.word_id_sequence_column]
    words = row[cfg.word_sequence_column]
    
    encoding = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
    input_ids = encoding["input_ids"]
    offset_mapping = encoding["offset_mapping"][0].tolist()
    
    if input_ids.shape[1] > cfg.max_tokens:
        input_ids = input_ids[:, :cfg.max_tokens]
        offset_mapping = offset_mapping[:cfg.max_tokens]
    
    with torch.no_grad():
        outputs = model(input_ids.to(device))
        logits = outputs.logits.cpu()
        log_probs_all = torch.log_softmax(logits, dim=-1)
        
    token_log_probs = [0.0] * input_ids.shape[1]
    for t_idx in range(1, input_ids.shape[1]):
        token_log_probs[t_idx] = log_probs_all[0, t_idx-1, input_ids[0, t_idx]].item()
        
    page_surprisals = []
    char_cursor = 0
    
    for word in words:
        word_start = char_cursor
        word_end = char_cursor + len(str(word))
        
        overlapping_log_probs = []
        for t_idx, (t_start, t_end) in enumerate(offset_mapping):
            if t_start < word_end and t_end > word_start:
                overlapping_log_probs.append(token_log_probs[t_idx])
        
        if not overlapping_log_probs:
            page_surprisals.append(float('nan'))
        else:
            page_surprisals.append(max(0.0, -sum(overlapping_log_probs)))
            
        char_cursor += len(str(word)) + 1
        
    page_surprisals = np.array(page_surprisals)
    mask = np.isnan(page_surprisals)
    n_truncated = mask.sum()
    if mask.any():
        nan_count += n_truncated
        mean_val = np.nanmean(page_surprisals) if not mask.all() else 0.0
        page_surprisals[mask] = mean_val
        
    for w_id, s_val in zip(word_ids, page_surprisals):
        results.append({'word_id': w_id, 'contextual_surprisal': s_val})
        
    if not tqdm and (i + 1) % 50 == 0:
        print(f"Processed {i + 1}/{total_pages} pages — page mean surprisal: {np.mean(page_surprisals):.3f} — truncated words: {n_truncated}")
        
    if (i + 1) % cfg.cache_clear_every_n_pages == 0:
        del outputs, logits, log_probs_all
        torch.cuda.empty_cache()

output_df = pd.DataFrame(results)
output_df.to_csv(os.path.join(cfg.output_dir, cfg.surprisal_output_filename), index=False)

print(f"{len(output_df)}")
print(f"{output_df['contextual_surprisal'].mean():.3f}, {output_df['contextual_surprisal'].std():.3f}")
print(f"{output_df['contextual_surprisal'].min():.3f}, {output_df['contextual_surprisal'].max():.3f}")
print(f"{nan_count}")

word_id_map_df = pd.read_csv(os.path.join(cfg.output_dir, "word_id_map.csv"))
if len(output_df) != len(word_id_map_df):
    print(f"WARNING: word_id count mismatch. Expected {len(word_id_map_df)}, got {len(output_df)}")
if output_df['word_id'].duplicated().sum() > 0:
    print(f"WARNING: {output_df['word_id'].duplicated().sum()} duplicate word_ids found in surprisal_features.csv")
if output_df['contextual_surprisal'].isnull().sum() > 0:
    print(f"WARNING: {output_df['contextual_surprisal'].isnull().sum()} NaN values remain in contextual_surprisal after imputation")
if (output_df['contextual_surprisal'] < 0).sum() > 0:
    print(f"WARNING: {(output_df['contextual_surprisal'] < 0).sum()} negative surprisal values found — check max(0.0, ...) clipping")
if len(output_df) == len(word_id_map_df) and output_df['word_id'].duplicated().sum() == 0 and output_df['contextual_surprisal'].isnull().sum() == 0 and (output_df['contextual_surprisal'] < 0).sum() == 0:
    print("Validation passed.")
