import pandas as pd
import re
from wordfreq import zipf_frequency

def extract_lexical_features(df):
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

    results = []
    
    for _, row in df.iterrows():
        word = str(row['word'])
        word_clean = re.sub(r'[^\w\s]', '', word)
        
        feat = {}
        feat['word_id'] = row['word_id']
        feat['word_len'] = len(word)
        feat['word_len_sq'] = len(word) ** 2
        feat['n_diacritics'] = len(re.findall(r'[ăâîșțĂÂÎȘȚ]', word))
        feat['is_capitalized'] = 1 if word and word[0].isupper() else 0
        feat['starts_sentence'] = 1 if row['word_id'].split('_')[-1] == '0' else 0
        feat['ends_punct'] = 1 if word and word[-1] in '.,!?;:' else 0
        feat['has_non_alpha'] = 1 if re.search(r'[^a-zA-ZăâîșțĂÂÎȘȚ]', word) else 0
        feat['zipf_frequency'] = zipf_frequency(word_clean, 'ro')
        feat['n_syllables'] = count_syllables(word_clean)
        feat['vowel_consonant_ratio'] = get_vowel_ratio(word_clean)
        
        romanian_stopwords = {"și", "la", "un", "o", "de", "pe", "cu", "din", "al", "a", "ai", "ale", "este", "sunt"}
        feat['is_stopword'] = 1 if word_clean.lower() in romanian_stopwords else 0
        
        results.append(feat)
        
    return pd.DataFrame(results)

train_df = pd.read_csv('../data/train_data.csv')
test_df = pd.read_csv('../data/test_data.csv')

train_lexical = extract_lexical_features(train_df)
test_lexical = extract_lexical_features(test_df)

train_lexical.to_csv('./output/train_features.csv', index=False)
test_lexical.to_csv('./output/test_features.csv', index=False)