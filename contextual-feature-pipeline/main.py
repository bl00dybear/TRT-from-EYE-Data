import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import re
import json
import subprocess

import numpy as np
import pandas as pd
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(SCRIPT_DIR, "../data/train_data.csv")
TEST_PATH = os.path.join(SCRIPT_DIR, "../data/test_data.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

SPACY_MODEL = "ro_core_news_lg"
LM_MODEL = "readerbench/RoGPT2-medium"
MAX_TOKENS = 900

CONTENT_POS_TAGS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
CASE_ORDER = ("Nom", "Acc", "Dat", "Gen", "Voc")
VERB_FORM_ORDER = ("Fin", "Inf", "Part", "Ger", "Sup")

os.makedirs(OUTPUT_DIR, exist_ok=True)


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

    word_id_map = df[["word_id", "text_id", "page_num", "word_idx"]]
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


def extract_ner_pos(nlp, df):
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
                results.append({
                    "word_id": w_id,
                    "is_NE": is_ne,
                    "NE_type": tok.ent_type_ if tok.ent_type_ != "" else "O",
                    "POS_tag": tok.pos_,
                    "is_content_word": 1 if tok.pos_ in CONTENT_POS_TAGS else 0,
                })
            else:
                results.append({
                    "word_id": w_id,
                    "is_NE": 0,
                    "NE_type": "O",
                    "POS_tag": "X",
                    "is_content_word": 0,
                })

    output_df = pd.DataFrame(results)
    output_df.to_csv(os.path.join(OUTPUT_DIR, "ner_pos_features.csv"), index=False)
    return output_df


def extract_surprisal(df):
    tokenizer = AutoTokenizer.from_pretrained(LM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LM_MODEL)
    model.eval()
    device = torch.device("cuda")
    model.to(device)

    results = []

    for i in range(len(df)):
        row = df.iloc[i]
        text = row["reconstructed_text"]
        word_ids = row["word_id_sequence"]
        words = row["word_sequence"]

        encoding = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
        input_ids = encoding["input_ids"]
        offset_mapping = encoding["offset_mapping"][0].tolist()

        if input_ids.shape[1] > MAX_TOKENS:
            input_ids = input_ids[:, :MAX_TOKENS]
            offset_mapping = offset_mapping[:MAX_TOKENS]

        with torch.no_grad():
            logits = model(input_ids.to(device)).logits.cpu()
            log_probs_all = torch.log_softmax(logits, dim=-1)

        token_log_probs = [0.0] * input_ids.shape[1]
        for t_idx in range(1, input_ids.shape[1]):
            token_log_probs[t_idx] = log_probs_all[0, t_idx - 1, input_ids[0, t_idx]].item()

        del logits, log_probs_all

        char_cursor = 0
        page_surprisals = []

        for word in words:
            word_start = char_cursor
            word_end = char_cursor + len(str(word))

            overlapping = []
            for t_idx, (t_start, t_end) in enumerate(offset_mapping):
                if t_start < word_end and t_end > word_start:
                    overlapping.append(token_log_probs[t_idx])

            if not overlapping:
                page_surprisals.append(float("nan"))
            else:
                page_surprisals.append(max(0.0, -sum(overlapping)))

            char_cursor += len(str(word)) + 1

        page_surprisals = np.array(page_surprisals)
        mask = np.isnan(page_surprisals)
        if mask.any():
            mean_val = np.nanmean(page_surprisals) if not mask.all() else 0.0
            page_surprisals[mask] = mean_val

        for w_id, s_val in zip(word_ids, page_surprisals):
            results.append({"word_id": w_id, "contextual_surprisal": s_val})

        if (i + 1) % 100 == 0:
            torch.cuda.empty_cache()

    output_df = pd.DataFrame(results)
    output_df.to_csv(os.path.join(OUTPUT_DIR, "surprisal_features.csv"), index=False)
    return output_df


def extract_morph(nlp, df):
    results = []

    for i in range(len(df)):
        row = df.iloc[i]
        word_ids = row["word_id_sequence"]
        words = row["word_sequence"]
        text = row["reconstructed_text"]

        doc = nlp(text)
        word_spans = build_word_spans(words, word_ids)
        mapping = align_tokens_to_words(doc, word_spans, word_ids)

        for w_id in word_ids:
            tokens = mapping[w_id]
            if tokens:
                tok = tokens[0]
                morph = tok.morph

                case_list = morph.get("Case")
                if not case_list:
                    case_encoded = -1
                else:
                    case_encoded = max(
                        CASE_ORDER.index(c) if c in CASE_ORDER else -1 for c in case_list
                    )

                vf_list = morph.get("VerbForm")
                if not vf_list:
                    verb_form = -1
                else:
                    verb_form = max(
                        VERB_FORM_ORDER.index(v) if v in VERB_FORM_ORDER else -1 for v in vf_list
                    )

                results.append({
                    "word_id": w_id,
                    "morph_count": len(morph),
                    "case_encoded": case_encoded,
                    "verb_form": verb_form,
                    "is_inflected": 1 if tok.text.lower().strip() != tok.lemma_.lower().strip() else 0,
                })
            else:
                results.append({
                    "word_id": w_id,
                    "morph_count": 0,
                    "case_encoded": -1,
                    "verb_form": -1,
                    "is_inflected": 0,
                })

    output_df = pd.DataFrame(results)
    output_df.to_csv(os.path.join(OUTPUT_DIR, "morph_features.csv"), index=False)
    return output_df


def aggregate():
    word_id_map = pd.read_csv(os.path.join(OUTPUT_DIR, "word_id_map.csv"))
    ner_pos = pd.read_csv(os.path.join(OUTPUT_DIR, "ner_pos_features.csv"))
    surprisal = pd.read_csv(os.path.join(OUTPUT_DIR, "surprisal_features.csv"))
    morph = pd.read_csv(os.path.join(OUTPUT_DIR, "morph_features.csv"))
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    features = word_id_map.merge(ner_pos, on="word_id", how="left")
    features = features.merge(surprisal, on="word_id", how="left")
    features = features.merge(morph, on="word_id", how="left")

    features["POS_tag"] = features["POS_tag"].fillna("X")
    features["NE_type"] = features["NE_type"].fillna("O")

    pos_map = {v: i for i, v in enumerate(sorted(features["POS_tag"].astype(str).unique()))}
    ne_map = {v: i for i, v in enumerate(sorted(features["NE_type"].astype(str).unique()))}
    features["POS_tag"] = features["POS_tag"].astype(str).map(pos_map)
    features["NE_type"] = features["NE_type"].astype(str).map(ne_map)

    train_features = train_df.merge(features, on="word_id", how="left")
    test_features = test_df.merge(features, on="word_id", how="left")

    train_features.to_csv(os.path.join(OUTPUT_DIR, "train_features.csv"), index=False)
    test_features.to_csv(os.path.join(OUTPUT_DIR, "test_features.csv"), index=False)


def main():
    if not spacy.util.is_package(SPACY_MODEL):
        subprocess.run(["python", "-m", "spacy", "download", SPACY_MODEL], check=True)

    spacy.prefer_gpu()
    nlp = spacy.load(SPACY_MODEL)

    reconstruct()
    df = load_pages()

    extract_ner_pos(nlp, df)
    extract_surprisal(df)
    extract_morph(nlp, df)
    aggregate()


if __name__ == "__main__":
    main()