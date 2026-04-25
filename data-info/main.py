import pandas as pd

train_df = pd.read_csv('../data/train_data.csv')
test_df = pd.read_csv('../data/test_data.csv')

train_ids = set(train_df['participant_id'].unique())
test_ids = set(test_df['participant_id'].unique())
all_ids = train_ids.union(test_ids)

print(f"Număr participanți unici în train: {len(train_ids)}")
print(f"Număr participanți unici în test: {len(test_ids)}")
print(f"Număr total participanți unici: {len(all_ids)}")