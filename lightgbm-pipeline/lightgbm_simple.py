import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from config_model import ModelConfig

def prepare_features(cfg, train_df, test_df):
    cols_to_drop = [c for c in cfg.drop_cols if c in train_df.columns]
    if cfg.target_col in train_df.columns:
        cols_to_drop.append(cfg.target_col)
    X_train_full = train_df.drop(columns=cols_to_drop)
    y_train_full = train_df[cfg.target_col]
    common_cols = [c for c in cfg.numeric_features if c in X_train_full.columns and c in test_df.columns]
    X_train = X_train_full[common_cols]
    X_test = test_df[common_cols]
    return X_train, y_train_full, X_test, common_cols

def split_data(cfg, train_df, X_train, y_train_full):
    unique_ids = train_df[cfg.participant_col].unique()
    np.random.seed(cfg.random_state)
    np.random.shuffle(unique_ids)
    n_val = int(len(unique_ids) * cfg.validation_participant_fraction)
    val_participants = unique_ids[-n_val:]
    val_mask = train_df[cfg.participant_col].isin(val_participants)
    X_tr = X_train[~val_mask]
    y_tr = y_train_full[~val_mask]
    X_val = X_train[val_mask]
    y_val = y_train_full[val_mask]
    return X_tr, y_tr, X_val, y_val

def train_lightgbm(cfg, X_tr, y_tr, X_val, y_val):
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
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
    callbacks = [lgb.early_stopping(cfg.early_stopping_rounds), lgb.log_evaluation(cfg.verbose_eval)]
    model = lgb.train(params, train_data, num_boost_round=cfg.n_estimators, valid_sets=[val_data], callbacks=callbacks)
    return model, params

def evaluate_and_save(cfg, model, params, X_val, y_val):
    val_preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    r2 = max(0, r2_score(y_val, val_preds))
    pearson = abs(pearsonr(y_val, val_preds)[0])
    score = 100 * (pearson + r2) / 2
    importances = model.feature_importance(importance_type='gain')
    feature_names = model.feature_name()
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
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
    print("\n=== VALIDATION METRICS ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Pearson: {pearson:.4f}")
    print(f"Hackathon Score: {score:.4f}")
    print("\n=== TOP 20 FEATURES (gain) ===")
    for name, imp in feat_imp[:20]:
        print(f"{name}: {imp:.4f}")

def create_submission(cfg, model, test_df, X_test):
    test_preds = model.predict(X_test)
    test_preds = np.clip(test_preds, 0, None)
    sub_id_col = cfg.submission_id_col if cfg.submission_id_col in test_df.columns else cfg.id_col
    datapoint_ids = test_df[sub_id_col].values
    submission_df = pd.DataFrame({
        "subtaskID": np.ones(len(test_df), dtype=int),
        "datapointID": datapoint_ids,
        "answer": test_preds
    })
    submission_df["datapointID"] = pd.to_numeric(submission_df["datapointID"], errors="coerce")
    submission_df = submission_df.sort_values("datapointID").reset_index(drop=True)
    sub_path = os.path.join(cfg.output_dir, cfg.submission_filename)
    submission_df.to_csv(sub_path, index=False)
    print("\n=== SUBMISSION SUMMARY ===")
    print(f"Rows: {len(submission_df)}")
    print(f"Predicted TRT — mean: {test_preds.mean():.2f}, std: {test_preds.std():.2f}, min: {test_preds.min():.2f}, max: {test_preds.max():.2f}")
    print(f"Saved to: {sub_path}")

def main():
    cfg = ModelConfig()
    train_df = pd.read_csv(cfg.train_path)
    test_df = pd.read_csv(cfg.test_path)
    X_train, y_train_full, X_test, common_cols = prepare_features(cfg, train_df, test_df)
    print(f"Feature columns: {sorted(common_cols)}")
    X_tr, y_tr, X_val, y_val = split_data(cfg, train_df, X_train, y_train_full)
    model, params = train_lightgbm(cfg, X_tr, y_tr, X_val, y_val)
    evaluate_and_save(cfg, model, params, X_val, y_val)
    create_submission(cfg, model, test_df, X_test)

if __name__ == "__main__":
    main()
