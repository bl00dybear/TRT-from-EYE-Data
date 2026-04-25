import spacy
nlp = spacy.load("ro_core_news_lg")
doc = nlp("Luna orbitează în jurul Pământului")
for token in doc:
    print(token.text, token.morph)