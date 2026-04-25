Ok, de la zero, simplu.

---

## Ce avem

Un CSV cu ~135k rânduri. Fiecare rând = un cuvânt citit de un participant, cu TRT-ul lui. Vrem să prezicem TRT pentru participanți noi pe aceleași cuvinte.

---

## De ce avem nevoie

Modelul (LightGBM) are nevoie de **features numerice** per cuvânt. Cu cât features-urile captează mai bine "dificultatea" unui cuvânt, cu atât prezicerea e mai bună.

Avem 3 categorii de features:

---

### Categoria 1 — Ce extragem direct din cuvântul scris
Fără GPU, fără modele. Pur string processing.

- Lungimea cuvântului
- Numărul de silabe
- Este URL? Are cifre? E cu majusculă?
- Poziția lui în pagină (word_idx / total_cuvinte_pagina)

**Acestea le face A, durează 1h.**

---

### Categoria 2 — Ce extragem din statisticile din train
Agregăm datele existente ca să înțelegem comportamentul mediu al fiecărui cuvânt.

- Media TRT a acestui cuvânt în train (peste toți participanții văzuți)
- Rata de skip a acestui cuvânt (câți participanți l-au sărit)
- Media TRT per text (textele enciclopedice sunt mai grele)

**Logica:** dacă "singurul" a avut mereu TRT=297ms în train, probabil va fi ~297ms și pentru participanți noi.

**Acestea le face A, tot în aceeași oră.**

---

### Categoria 3 — Ce extragem cu modele NLP pe GPU
Aici e valoarea adăugată față de soluțiile simple.

**NER** — este cuvântul un named entity?
- "Luna" ca satelit vs "Luna" ca nume propriu — NER le distinge
- Named entities au TRT mai mare (mai puțin predictibile)

**POS tagging** — ce parte de vorbire e cuvântul?
- Substantivele și verbele se citesc mai lent decât articolele și prepozițiile
- "este" (verb funcțional) vs "orbitează" (verb content) — diferite

**Surprizal contextual** — cât de surprinzător e cuvântul dat contextul anterior?
- `-log P(cuvânt | tot ce a venit înainte)`
- "Luna orbitează în jurul **Pământului**" — Pământului e predictibil, surprizal mic, TRT mic
- "Luna orbitează în jurul **Soarelui**" — surprizal mare, TRT mare
- Asta necesită GPT2 românesc care citește fraza și estimează probabilitatea fiecărui cuvânt

**Acestea le face B pe GPU, durează 3-4h.**

---

## Cum se asamblează

```
word_id → features categoria 1 (A)
word_id → features categoria 2 (A)
word_id → features categoria 3 (B)
         ↓
    merge pe word_id
         ↓
    feature matrix completă
         ↓
    LightGBM → prezice TRT
         ↓
    submission.csv
```

---

## De ce participanții noi nu sunt o problemă

Nu folosim nimic specific participantului (nu există `participant_mean_speed` sau similar). Toate features-urile sunt despre **cuvânt** și **context textual**. Asta înseamnă că prezicerea pentru un participant nou e identică cu orice alt participant nou — prezici TRT-ul "mediu al populației" pentru acel cuvânt.

---

## Ce nu facem și de ce

**Nu facem embedding-uri dense** (e.g., media vectorilor BERT per cuvânt) — LightGBM nu lucrează bine cu 768 features dense corelate, și nu avem timp de fine-tuning.

**Nu facem hurdle acum** — mai întâi submission simplu, dacă scorul e slab și vedem că 0-urile strică, adăugăm hurdle în iterația 2.

---

Clar acum? Sau e vreo parte specifică pe care vrei să o aprofundăm?