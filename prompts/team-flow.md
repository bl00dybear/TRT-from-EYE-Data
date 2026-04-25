**Asumpție:** tu + un coleg, ambii pe aceeași mașină cu cele 2 GPU-uri.

---

## Persoana A — Feature Engineering + Model

### Ce face:
Tot ce e CPU-based + modelul final.

**Task 1 — EDA + setup (1h)**
- Încarci train.csv, test.csv
- Verifici distribuția TRT, câte 0-uri, câți participanți, câte texte unice
- Reconstruiești textele (groupby text_id + page, sort word_idx, join) și le salvezi într-un fișier separat — **B are nevoie de asta**
- Stabilești coloanele din `word_id` (corpus, page, word_idx)

**Task 2 — Feature engineering lexical (1-2h)**
- word_len, word_len_sq, n_vowels, n_syllables, vowel_ratio
- is_capitalized, is_all_upper, has_digit, has_hyphen, is_url, ends_punct
- word_idx, page_num, rel_position
- unigram_surprisal din train (simplu, fără GPU)
- aggregate stats per word: mean_trt, median_trt, std_trt, skip_rate
- aggregate stats per text_id: mean_trt, std_trt

**Task 3 — Așteaptă outputul de la B (async)**
- Când B termină NER și surprizal contextual, le mergi pe `word_id`
- Adaugi coloanele: `is_NE`, `NE_type`, `contextual_surprisal`, `POS_tag`, `is_content_word`

**Task 4 — Model (2-3h)**
- LightGBM regressor pe toate datele (inclusiv 0-uri)
- Validare: split pe `participant_id` — nu random split, că altfel ai leak
- Optimizezi hyperparametri cu Optuna
- Opțional dacă timpul permite: hurdle (XGB classifier skip/no-skip + LightGBM pe TRT>0)

**Task 5 — Submission**
- Generezi submission.csv în formatul cerut

---

## Persoana B — NLP / GPU Features

### Ce face:
Tot ce rulează pe GPU. Primește textele reconstruite de la A.

**Task 1 — Setup environment (30min)**
```
pip install transformers torch spacy
```
- GPU 0 → NER
- GPU 1 → Surprizal GPT2

**Task 2 — NER pe GPU 0 (2-3h)**
- Model: `dumitrescuvalentin/bert-base-romanian-cased-ner`
- Input: textele reconstruite per pagină
- Output per cuvânt: `is_NE` (0/1), `NE_type` (PER/LOC/ORG/MISC/O)
- Salvezi un CSV cu coloanele `word_id`, `is_NE`, `NE_type`
- Atenție la alinierea token→cuvânt (BPE): pentru un cuvânt cu mai mulți subtokeni, iei label-ul primului subtoken (convenție standard)

**Task 3 — POS tagging (poate fi combinat cu NER, același model sau spaCy)**
- Dacă modelul NER nu dă și POS, folosești `spacy` cu modelul `ro_core_news_lg`
- Output per cuvânt: `POS_tag`, `is_content_word` (NOUN/VERB/ADJ/ADV = 1, restul = 0)
- Adaugi la același CSV de output

**Task 4 — Surprizal contextual pe GPU 1 (3-4h)**
- Model: `readerbench/ro-gpt2`
- Pentru fiecare pagină reconstruită, procesezi secvențial cuvintele
- Context per cuvânt: fraza anterioară + fraza curentă până la cuvântul respectiv
- Extragi log P(cuvânt | context) — pentru cuvinte multi-token: suma log-prob a subtokenilor
- Output: CSV cu `word_id`, `contextual_surprisal`

**Task 5 — Merge și trimite la A**
- Mergi NER + POS + surprizal pe `word_id`
- Un singur fișier: `nlp_features.csv`

---

## Flow-ul de lucru

```
T=0h    A: EDA + reconstruire texte ──────────────────► B primește textele
        B: setup environment

T=1h    A: feature engineering lexical (independent)
        B: NER pe GPU 0 + POS ────────────────────────► paralel

T=2h    A: continuă features + începe să scrie
        B: surprizal pe GPU 1 ────────────────────────► paralel cu NER
        pipeline-ul de model

T=4h    B: trimite nlp_features.csv la A ─────────────► A merge totul

T=5h    A: model complet cu toate features
            + validare + hyperparameter tuning

T=6h    primul submission pe leaderboard
            ────────────────────────────────────────────► vedeți scorul

T=6-24h iterații: features noi, hurdle, eroare analysis
```

---

## Puncte de sincronizare critice

1. **T=0** — A salvează textele reconstruite într-un format clar (CSV cu `text_id`, `page_num`, `reconstructed_text`). B nu poate începe fără asta.
2. **T=4** — B livrează `nlp_features.csv` cu exact coloanele `word_id`, `is_NE`, `NE_type`, `POS_tag`, `is_content_word`, `contextual_surprisal`. Schema trebuie agreată acum, nu la T=4.
3. **Validarea** — amândoi folosiți același split de validare ca să comparați scorurile corect.

---

## Risc principal

Dacă surprizalul GPT2 durează mai mult decât estimat, B poate livra NER+POS mai întâi (T=3h) și A face primul submission fără surprizal. Surprizalul vine în iterația a doua.