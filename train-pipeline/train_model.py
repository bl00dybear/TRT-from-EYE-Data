import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from config_model import ModelConfig

cfg = ModelConfig()

assert cfg.train_path != "", "train_path is not set in config_model.py"
assert cfg.test_path != "", "test_path is not set in config_model.py"

train_df = pd.read_csv(cfg.train_path)
test_df = pd.read_csv(cfg.test_path)

for df in [train_df, test_df]:
    if "text_id" not in df.columns and cfg.id_col in df.columns:
        df["text_id"] = df[cfg.id_col].str.extract(r'^(.*)_page_\d+_\d+$')

for col in cfg.categorical_features:
    if col in train_df.columns:
        train_df[col] = train_df[col].astype("category")
        if col in test_df.columns:
            test_df[col] = pd.Categorical(test_df[col], categories=train_df[col].cat.categories)

cols_to_drop = [c for c in cfg.drop_cols if c in train_df.columns]
if cfg.target_col in train_df.columns:
    cols_to_drop.append(cfg.target_col)

X_train_full = train_df.drop(columns=cols_to_drop)
y_train_full = train_df[cfg.target_col]

common_cols = list(set(X_train_full.columns).intersection(test_df.columns))
X_train = X_train_full[common_cols]
X_test = test_df[common_cols]

print(f"Feature columns: {sorted(common_cols)}")

unique_ids = train_df[cfg.participant_col].unique()
np.random.seed(cfg.random_state)
np.random.shuffle(unique_ids)

n_val = int(len(unique_ids) * cfg.validation_participant_fraction)
val_participants = unique_ids[-n_val:]
train_participants = unique_ids[:-n_val]

val_mask = train_df[cfg.participant_col].isin(val_participants)

X_tr, y_tr = X_train[~val_mask], y_train_full[~val_mask]
X_val, y_val = X_train[val_mask], y_train_full[val_mask]

print(f"Train: {len(train_participants)} participants, {len(X_tr)} rows")
print(f"Validation: {len(val_participants)} participants, {len(X_val)} rows")

train_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=[c for c in cfg.categorical_features if c in X_tr.columns])
val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=[c for c in cfg.categorical_features if c in X_val.columns], reference=train_data)

params = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": cfg.num_leaves,
    "learning_rate": cfg.learning_rate,
    "min_child_samples": cfg.min_child_samples,
    "subsample": cfg.subsample,
    "colsample_bytree": cfg.colsample_bytree,
    "reg_alpha": cfg.reg_alpha,
    "reg_lambda": cfg.reg_lambda,
    "random_state": cfg.random_state,
    "verbose": -1,
}

callbacks = [
    lgb.early_stopping(cfg.early_stopping_rounds),
    lgb.log_evaluation(cfg.verbose_eval)
]

model = lgb.train(
    params,
    train_data,
    num_boost_round=cfg.n_estimators,
    valid_sets=[val_data],
    callbacks=callbacks,
)

val_preds = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
r2 = max(0, r2_score(y_val, val_preds))
pearson = abs(pearsonr(y_val, val_preds)[0])
score = 100 * (pearson + r2) / 2

print("\n=== VALIDATION METRICS ===")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Pearson: {pearson:.4f}")
print(f"Hackathon Score: {score:.4f}")

importances = model.feature_importance(importance_type='gain')
feature_names = model.feature_name()
feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

print("\n=== TOP 20 FEATURES (gain) ===")
for name, imp in feat_imp[:20]:
    print(f"{name}: {imp:.4f}")

os.makedirs(cfg.output_dir, exist_ok=True)
model_path = os.path.join(cfg.output_dir, cfg.model_filename)
model.save_model(model_path)

report_path = os.path.join(cfg.output_dir, cfg.validation_report_filename)
with open(report_path, "w") as f:
    f.write("=== VALIDATION METRICS ===\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"R2: {r2:.4f}\n")
    f.write(f"Pearson: {pearson:.4f}\n")
    f.write(f"Hackathon Score: {score:.4f}\n\n")
    f.write("=== TOP 20 FEATURES (gain) ===\n")
    for name, imp in feat_imp[:20]:
        f.write(f"{name}: {imp:.4f}\n")
    f.write(f"\nTraining params: {params}\n")
    f.write(f"Best iteration: {model.best_iteration}\n")

test_preds = model.predict(X_test)
test_preds = np.clip(test_preds, 0, None)

sub_id_col = cfg.submission_id_col if cfg.submission_id_col in test_df.columns else cfg.id_col
submission_df = pd.DataFrame({
    sub_id_col: test_df[sub_id_col],
    "answer": test_preds
})

sub_path = os.path.join(cfg.output_dir, cfg.submission_filename)
submission_df.to_csv(sub_path, index=False)

print("\n=== SUBMISSION SUMMARY ===")
print(f"Rows: {len(submission_df)}")
print(f"Predicted TRT — mean: {test_preds.mean():.2f}, std: {test_preds.std():.2f}, min: {test_preds.min():.2f}, max: {test_preds.max():.2f}")
print(f"Saved to: {sub_path}")
