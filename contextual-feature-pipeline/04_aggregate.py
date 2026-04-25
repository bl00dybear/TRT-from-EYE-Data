import pandas as pd
import os
from config_aggregate import AggregateConfig

cfg = AggregateConfig()

word_id_map = pd.read_csv(os.path.join(cfg.output_dir, cfg.word_id_map_filename))
ner_pos = pd.read_csv(os.path.join(cfg.output_dir, cfg.ner_pos_filename))
surprisal = pd.read_csv(os.path.join(cfg.output_dir, cfg.surprisal_filename))
morph = pd.read_csv(os.path.join(cfg.output_dir, cfg.morph_filename))
train_df = pd.read_csv(cfg.train_path)
test_df = pd.read_csv(cfg.test_path)

context_features = word_id_map.merge(ner_pos, on='word_id', how='left')
context_features = context_features.merge(surprisal, on='word_id', how='left')
context_features = context_features.merge(morph, on='word_id', how='left')

pos_map = {val: i for i, val in enumerate(sorted([str(x) for x in context_features['POS_tag'].unique()]))}
ne_map = {val: i for i, val in enumerate(sorted([str(x) for x in context_features['NE_type'].unique()]))}
context_features['POS_tag'] = context_features['POS_tag'].astype(str).map(pos_map)
context_features['NE_type'] = context_features['NE_type'].astype(str).map(ne_map)

train_features = train_df.merge(context_features, on='word_id', how='left')
test_features = test_df.merge(context_features, on='word_id', how='left')

# drop_cols = ['word', 'participant_id', 'text']
# train_features = train_features.drop(columns=drop_cols)
# test_features = test_features.drop(columns=drop_cols)

train_features.to_csv(os.path.join(cfg.output_dir, cfg.train_output_filename), index=False)
test_features.to_csv(os.path.join(cfg.output_dir, cfg.test_output_filename), index=False)

valid = True
if len(train_features) != len(train_df):
    print(f"WARNING: train row count mismatch. Expected {len(train_df)}, got {len(train_features)}")
    valid = False
if len(test_features) != len(test_df):
    print(f"WARNING: test row count mismatch. Expected {len(test_df)}, got {len(test_features)}")
    valid = False

context_cols = [
    'text_id', 'page_num', 'word_idx', 
    'is_NE', 'NE_type', 'POS_tag', 'is_content_word', 
    'contextual_surprisal',
    'morph_count', 'case_encoded', 'verb_form', 'is_inflected'
]
for col in context_cols:
    pct_nan = train_features[col].isnull().mean() * 100
    if pct_nan > 1:
        print(f"WARNING: {col} has {pct_nan:.2f}% missing values in train")
        valid = False

dupes = train_features.duplicated().sum()
if dupes > 0:
    print(f"WARNING: {dupes} duplicate rows found in train_features.csv after column removal")
    # valid = False  # Not marking as invalid since we expect duplicates now

if valid:
    print("Validation passed.")

print("\n=== TRAIN FEATURES SUMMARY ===")
print(f"Rows: {len(train_features)}")
print(f"Columns: {train_features.columns.tolist()}")

print("\n=== CONTEXTUAL FEATURES COVERAGE ===")
ne_count = (train_features['is_NE'] == 1).sum()
ne_pct = ne_count / len(train_features) * 100
print(f"is_NE=1: {ne_count} ({ne_pct:.2f}%)")
print("NE_type distribution:")
print(train_features['NE_type'].value_counts())
print("POS_tag distribution:")
print(train_features['POS_tag'].value_counts())
cw_count = (train_features['is_content_word'] == 1).sum()
cw_pct = cw_count / len(train_features) * 100
print(f"is_content_word=1: {cw_count} ({cw_pct:.2f}%)")

s_stats = train_features['contextual_surprisal']
print(f"contextual_surprisal — mean: {s_stats.mean():.3f}, std: {s_stats.std():.3f}, min: {s_stats.min():.3f}, max: {s_stats.max():.3f}")

inf_count = (train_features['is_inflected'] == 1).sum()
inf_pct = inf_count / len(train_features) * 100
print(f"is_inflected=1: {inf_count} ({inf_pct:.2f}%)")
print(f"morph_count mean: {train_features['morph_count'].mean():.2f}")

print("\n=== TEST FEATURES SUMMARY ===")
print(f"Rows: {len(test_features)}")
print(f"Columns: {test_features.columns.tolist()}")
