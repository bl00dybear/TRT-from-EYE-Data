import pandas as pd
import os

def load_and_merge(lex_path, ctx_path):
    df_lex = pd.read_csv(lex_path)
    df_ctx = pd.read_csv(ctx_path)
    merged = pd.merge(df_lex, df_ctx, on='word_id', how='left')
    if 'text_id' in merged.columns:
        merged = merged.drop(columns=['text_id'])
    return merged

train_df = load_and_merge(
    './lexical-feature-pipeline/output/train_features.csv',
    './contextual-feature-pipeline/output/train_features.csv'
)

test_df = load_and_merge(
    './lexical-feature-pipeline/output/test_features.csv',
    './contextual-feature-pipeline/output/test_features.csv'
)

os.makedirs('./merged_output', exist_ok=True)
train_df.to_csv('./merged_output/train_final.csv', index=False)
test_df.to_csv('./merged_output/test_final.csv', index=False)