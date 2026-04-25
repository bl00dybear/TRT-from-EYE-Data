import pandas as pd
import numpy as np

# 1. Load the files
df_lexical = pd.read_csv('./contextual-feature-pipeline/output/train_features.csv')
df_context = pd.read_csv('./lexical-feature-pipeline/output/train_features.csv')

# 2. Merge on word_id
# We use 'inner' to ensure we only keep rows present in both pipelines
train_final = pd.merge(df_lexical, df_context, on='word_id', how='inner')

# 3. Handle the 'answer' column (Target)
# If answer is TRT, log transform often improves Pearson correlation
train_final['target'] = np.log1p(train_final['answer'])

# 4. Final Feature Selection
# Drop columns that are strings or identifiers
X = train_final.drop(columns=['word_id', 'answer', 'target'])
y = train_final['target']

# 5. Define Categorical Indices for LightGBM
cat_features = ['text_id', 'POS_tag', 'NE_type', 'case_encoded', 'verb_form']