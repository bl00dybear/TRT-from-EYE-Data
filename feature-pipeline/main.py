import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import subprocess
import math
import numpy as np
import pandas as pd
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from wordfreq import zipf_frequency

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(SCRIPT_DIR, "../data/train_data.csv")
TEST_PATH = os.path.join(SCRIPT_DIR, "../data/test_data.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

SPACY_MODEL = "ro_core_news_lg"
LM_MODEL = "readerbench/RoGPT2-medium"
MAX_TOKENS = 1000
STRIDE = 512

CONTENT_POS_TAGS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
CASE_ORDER = ("Nom", "Acc", "Dat", "Gen", "Voc")
VERB_FORM_ORDER = ("Fin", "Inf", "Part", "Ger", "Sup")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def count_syllables(word):
    word = str(word).lower()
    vowels = "aeiouăâî"
    count = 0
    if not word: return 0
    if word[0] in vowels: count += 1
    for i in range(1, len(word)):
        if word[i] in vowels and word[i-1] not in vowels:
            count += 1
    return max(1, count)

def get_vowel_ratio(word):
    word = str(word).lower()
    vowels = "aeiouăâî"
    v_count = sum(1 for char in word if char in vowels)
    return v_count / len(word) if len(word) > 0 else 0

def extract_lexical_features(df, nlp):
    df_out = pd.DataFrame()
    df_out['word_id'] = df['word_id']
    words_str = df['word'].astype(str)
    words_clean = words_str.str.replace(r'[^\w\s]', '', regex=True)
    df_out['word_len'] = words_str.str.len()
    df_out['word_len_sq'] = df_out['word_len'] ** 2
    df_out['n_diacritics'] = words_str.str.count(r'[ăâîșțĂÂÎȘȚ]')
    df_out['is_capitalized'] = words_str.str[0].str.isupper().astype(int)
    df_out['starts_sentence'] = (df['word_id'].str.split('_').str[-1] == '0').astype(int)
    df_out['ends_punct'] = words_str.str[-1].isin(list('.,!?;:')).astype(int)
    df_out['has_non_alpha'] = words_str.str.contains(r'[^a-zA-ZăâîșțĂÂÎȘȚ]', regex=True).astype(int)
    df_out['zipf_frequency'] = words_clean.apply(lambda w: zipf_frequency(w, 'ro'))
    df_out['n_syllables'] = words_clean.apply(count_syllables)
    df_out['vowel_consonant_ratio'] = words_clean.apply(get_vowel_ratio)
    df_out['is_stopword'] = words_clean.str.lower().isin(nlp.Defaults.stop_words).astype(int)
    return df_out

def build_word_spans(word_list, word_id_list):
    spans = []
    cursor = 0
    for i, w in enumerate(word_list):
        w_text = str(w)
        spans.append((cursor, cursor + len(w_text), word_id_list[i]))
        cursor += len(w_text) + 1
    return spans

def align_tokens_to_words(doc, word_spans, word_id_list):
    mapping = {w_id: [] for w_id in word_id_list}
    for token in doc:
        if token.is_space:
            continue
        t_start = token.idx
        t_end = token.idx + len(token.text)
        for w_start, w_end, w_id in word_spans:
            if t_start < w_end and t_end > w_start:
                mapping[w_id].append(token)
                break
    return mapping

def reconstruct():
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    df = pd.concat([df_train, df_test]).drop_duplicates(subset=["word_id"], keep="first")
    
    parsed = df["word_id"].str.extract(r'^(.*)_page_(\d+)_(\d+)$')
    df["text_id"] = parsed[0]
    df["page_num"] = parsed[1].astype(int)
    df["word_idx"] = parsed[2].astype(int)
    
    word_id_map = df[["word_id", "word", "text_id", "page_num", "word_idx"]]
    word_id_map.to_csv(os.path.join(OUTPUT_DIR, "word_id_map.csv"), index=False)
    
    grouped = df.sort_values(["text_id", "page_num", "word_idx"]).groupby(["text_id", "page_num"])
    rows = []
    for (text_id, page_num), group in grouped:
        rows.append({
            "text_id": text_id,
            "page_num": page_num,
            "reconstructed_text": " ".join(group["word"].astype(str).tolist()),
            "word_id_sequence": json.dumps(group["word_id"].tolist()),
            "word_sequence": json.dumps(group["word"].tolist()),
        })
    
    reconstructed_df = pd.DataFrame(rows)
    reconstructed_df.to_csv(os.path.join(OUTPUT_DIR, "reconstructed_pages.csv"), index=False)
    return reconstructed_df

def load_pages():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "reconstructed_pages.csv"))
    df["word_id_sequence"] = df["word_id_sequence"].apply(json.loads)
    df["word_sequence"] = df["word_sequence"].apply(json.loads)
    return df

def extract_spacy_features(nlp, df):
    results = []
    for i, doc in enumerate(nlp.pipe(df["reconstructed_text"], batch_size=32)):
        row = df.iloc[i]
        word_ids = row["word_id_sequence"]
        words = row["word_sequence"]
        word_spans = build_word_spans(words, word_ids)
        mapping = align_tokens_to_words(doc, word_spans, word_ids)
        
        for w_id in word_ids:
            tokens = mapping[w_id]
            if tokens:
                tok = tokens[0]
                is_ne = 1 if tok.ent_type_ != "" else 0
                morph = tok.morph
                case_list = morph.get("Case")
                case_encoded = max(CASE_ORDER.index(c) if c in CASE_ORDER else -1 for c in case_list) if case_list else -1
                vf_list = morph.get("VerbForm")
                verb_form = max(VERB_FORM_ORDER.index(v) if v in VERB_FORM_ORDER else -1 for v in vf_list) if vf_list else -1
                
                ancestors = list(tok.ancestors)
                dep_depth = len(ancestors)
                
                results.append({
                    "word_id": w_id,
                    "is_NE": is_ne,
                    "NE_type": tok.ent_type_ if is_ne else "O",
                    "POS_tag": tok.pos_,
                    "dep_rel": tok.dep_,
                    "dep_depth": dep_depth,
                    "is_content_word": 1 if tok.pos_ in CONTENT_POS_TAGS else 0,
                    "morph_count": len(morph),
                    "case_encoded": case_encoded,
                    "verb_form": verb_form,
                    "is_inflected": 1 if tok.text.lower().strip() != tok.lemma_.lower().strip() else 0,
                })
            else:
                results.append({
                    "word_id": w_id,
                    "is_NE": 0,
                    "NE_type": "O",
                    "POS_tag": "X",
                    "dep_rel": "X",
                    "dep_depth": 0,
                    "is_content_word": 0,
                    "morph_count": 0,
                    "case_encoded": -1,
                    "verb_form": -1,
                    "is_inflected": 0,
                })
    output_df = pd.DataFrame(results)
    output_df.to_csv(os.path.join(OUTPUT_DIR, "spacy_features.csv"), index=False)
    return output_df

def extract_surprisal(df):
    tokenizer = AutoTokenizer.from_pretrained(LM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LM_MODEL)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    results = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        text = row["reconstructed_text"]
        word_ids = row["word_id_sequence"]
        words = row["word_sequence"]
        encoding = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
        input_ids = encoding["input_ids"][0]
        offset_mapping = encoding["offset_mapping"][0].tolist()
        seq_len = input_ids.size(0)
        token_log_probs = [0.0] * seq_len
        
        for start_idx in range(0, seq_len, STRIDE):
            end_idx = min(start_idx + MAX_TOKENS, seq_len)
            chunk_input_ids = input_ids[start_idx:end_idx].unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(chunk_input_ids).logits.cpu()
                log_probs_all = torch.log_softmax(logits, dim=-1)
            chunk_len = chunk_input_ids.size(1)
            for j in range(1, chunk_len):
                global_idx = start_idx + j
                if token_log_probs[global_idx] == 0.0:
                    token_log_probs[global_idx] = log_probs_all[0, j - 1, chunk_input_ids[0, j]].item() / math.log(2)
            if end_idx == seq_len:
                break
                
        char_cursor = 0
        for w_idx, word in enumerate(words):
            word_str = str(word)
            word_start = char_cursor
            word_end = char_cursor + len(word_str)
            overlapping = []
            for t_idx, (t_start, t_end) in enumerate(offset_mapping):
                if t_start < word_end and t_end > word_start:
                    overlapping.append(token_log_probs[t_idx])
            s_val = max(0.0, -sum(overlapping)) if overlapping else 0.0
            results.append({"word_id": word_ids[w_idx], "contextual_surprisal": s_val})
            char_cursor += len(word_str) + 1
            
        if (i + 1) % 100 == 0:
            torch.cuda.empty_cache()
            
    output_df = pd.DataFrame(results)
    output_df.to_csv(os.path.join(OUTPUT_DIR, "surprisal_features.csv"), index=False)
    return output_df

def create_parafoveal_features(df):
    df = df.sort_values(['text_id', 'page_num', 'word_idx'])
    for shift_val, name in zip([-1, 1], ['next', 'prev']):
        df[f'{name}_word_len'] = df.groupby(['text_id', 'page_num'])['word_len'].shift(shift_val).fillna(0)
        df[f'{name}_surprisal'] = df.groupby(['text_id', 'page_num'])['contextual_surprisal'].shift(shift_val).fillna(0)
        df[f'{name}_zipf'] = df.groupby(['text_id', 'page_num'])['zipf_frequency'].shift(shift_val).fillna(0)
    return df

def add_macro_cognitive_features(df):
    text_lengths = df.groupby(['text_id', 'page_num'])['word_id'].transform('count')
    df['relative_position'] = df['word_idx'] / text_lengths
    
    df['surprisal_freq_interaction'] = df['contextual_surprisal'] * np.log2(1 + df['zipf_frequency'])
    
    def calculate_ttr(group):
        words = group['word'].astype(str).str.lower()
        return len(words.unique()) / len(words)
        
    ttr_map = df.groupby('text_id').apply(calculate_ttr).reset_index(name='text_ttr')
    df = df.merge(ttr_map, on='text_id', how='left')
    return df

def aggregate(nlp):
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    word_id_map = pd.read_csv(os.path.join(OUTPUT_DIR, "word_id_map.csv"))
    
    lexical_feats = extract_lexical_features(word_id_map, nlp)
    spacy_feats = pd.read_csv(os.path.join(OUTPUT_DIR, "spacy_features.csv"))
    surprisal = pd.read_csv(os.path.join(OUTPUT_DIR, "surprisal_features.csv"))
    
    master_features = word_id_map.merge(lexical_feats, on="word_id", how="left")
    master_features = master_features.merge(spacy_feats, on="word_id", how="left")
    master_features = master_features.merge(surprisal, on="word_id", how="left")
    
    master_features["POS_tag"] = master_features["POS_tag"].fillna("X")
    master_features["NE_type"] = master_features["NE_type"].fillna("O")
    master_features["dep_rel"] = master_features["dep_rel"].fillna("X")
    
    pos_map = {v: i for i, v in enumerate(sorted(master_features["POS_tag"].astype(str).unique()))}
    ne_map = {v: i for i, v in enumerate(sorted(master_features["NE_type"].astype(str).unique()))}
    dep_map = {v: i for i, v in enumerate(sorted(master_features["dep_rel"].astype(str).unique()))}
    
    master_features["POS_tag"] = master_features["POS_tag"].astype(str).map(pos_map)
    master_features["NE_type"] = master_features["NE_type"].astype(str).map(ne_map)
    master_features["dep_rel"] = master_features["dep_rel"].astype(str).map(dep_map)
    
    master_features = create_parafoveal_features(master_features)
    master_features = add_macro_cognitive_features(master_features)
    
    drop_cols = ['word', 'text_id', 'page_num', 'word_idx']
    master_features = master_features.drop(columns=[c for c in drop_cols if c in master_features.columns])
    
    train_features = train_df.merge(master_features, on="word_id", how="left")
    test_features = test_df.merge(master_features, on="word_id", how="left")
    
    train_features.to_csv(os.path.join(OUTPUT_DIR, "train_features.csv"), index=False)
    test_features.to_csv(os.path.join(OUTPUT_DIR, "test_features.csv"), index=False)

def main():
    if not spacy.util.is_package(SPACY_MODEL):
        subprocess.run(["python", "-m", "spacy", "download", SPACY_MODEL], check=True)
    spacy.prefer_gpu()
    nlp = spacy.load(SPACY_MODEL)
    
    reconstruct()
    df = load_pages()
    
    extract_spacy_features(nlp, df)
    extract_surprisal(df)
    aggregate(nlp)

if __name__ == "__main__":
    main()