import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from config_model import ModelConfig

def set_category_columns(cfg, train_df, test_df):
    for c in cfg.categorical_cols:
        if c in train_df.columns:
            train_df[c] = train_df[c].astype('category')
        if c in test_df.columns:
            test_df[c] = test_df[c].astype('category')
    return train_df, test_df


def existing_cols(df, cols):
    return [col for col in cols if col in df.columns]


def sort_dataframes(train_df, test_df):
    sort_cols = existing_cols(train_df, ['participant_id', 'text', 'page_num', 'word_idx'])
    if len(sort_cols) > 0:
        train_df = train_df.sort_values(by=sort_cols)
        test_df = test_df.sort_values(by=sort_cols)
    return train_df, test_df


def add_lag_lead_columns(cfg, train_df, test_df):
    new_lag_lead_cols = []
    group_cols = existing_cols(train_df, ['participant_id', 'text'])
    
    if len(group_cols) == 0:
        group_cols = existing_cols(train_df, ['participant_id'])
    
    for col in cfg.cols_to_shift:
        if col in train_df.columns:
            lag1, lead1 = f'{col}_lag1', f'{col}_lead1'
            lag2, lead2 = f'{col}_lag2', f'{col}_lead2'
            
            train_df[lag1] = train_df.groupby(group_cols)[col].shift(1).fillna(0)
            train_df[lead1] = train_df.groupby(group_cols)[col].shift(-1).fillna(0)
            train_df[lag2] = train_df.groupby(group_cols)[col].shift(2).fillna(0)
            train_df[lead2] = train_df.groupby(group_cols)[col].shift(-2).fillna(0)
            
            test_df[lag1] = test_df.groupby(group_cols)[col].shift(1).fillna(0)
            test_df[lead1] = test_df.groupby(group_cols)[col].shift(-1).fillna(0)
            test_df[lag2] = test_df.groupby(group_cols)[col].shift(2).fillna(0)
            test_df[lead2] = test_df.groupby(group_cols)[col].shift(-2).fillna(0)
            
            new_lag_lead_cols.extend([lag1, lead1, lag2, lead2])
            
    return train_df, test_df, new_lag_lead_cols


def get_feature_columns(cfg, train_df, test_df, new_lag_lead_cols):
    cols_to_drop = [c for c in cfg.drop_cols if c in train_df.columns]
    if cfg.target_col in train_df.columns:
        cols_to_drop.append(cfg.target_col)
    if cfg.participant_col in train_df.columns and cfg.participant_col not in cols_to_drop:
        cols_to_drop.append(cfg.participant_col)

    X_train_full = train_df.drop(columns=cols_to_drop)
    
    common_cols = [c for c in cfg.numeric_features if c in X_train_full.columns and c in test_df.columns]
    
    for c in new_lag_lead_cols + cfg.categorical_cols:
        if c not in common_cols and c in X_train_full.columns and c in test_df.columns:
            common_cols.append(c)

    return list(set(common_cols) - set(cols_to_drop)), cols_to_drop



def split_validation_data(cfg, train_df, X_train, y_train_full):
    unique_ids = np.array(train_df[cfg.participant_col].unique())
    np.random.seed(cfg.random_state)
    np.random.shuffle(unique_ids)

    n_val = int(len(unique_ids) * cfg.validation_participant_fraction)
    val_participants = unique_ids[-n_val:]
    
    val_mask = train_df[cfg.participant_col].isin(val_participants)
    
    X_tr, y_tr = X_train[~val_mask], y_train_full[~val_mask]
    X_val, y_val = X_train[val_mask], y_train_full[val_mask]
    
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

    callbacks = [
        lgb.early_stopping(cfg.early_stopping_rounds),
        lgb.log_evaluation(10)
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=cfg.n_estimators,
        valid_sets=[val_data],
        callbacks=callbacks,
    )
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
        
    print(f"RMSE: {rmse:.4f} | R^2: {r2:.4f} | Pearson: {pearson:.4f} | Score: {score:.4f}")

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
    print(f"Saved submission to: {sub_path}")

def main():
    cfg = ModelConfig()
    
    train_df = pd.read_csv(cfg.train_path)
    test_df = pd.read_csv(cfg.test_path)
    
    train_df, test_df = set_category_columns(cfg, train_df, test_df)
    train_df, test_df = sort_dataframes(train_df, test_df)
    train_df, test_df, new_lag_lead_cols = add_lag_lead_columns(cfg, train_df, test_df)
    
    common_cols, cols_to_drop = get_feature_columns(cfg, train_df, test_df, new_lag_lead_cols)
    
    X_train_full = train_df.drop(columns=cols_to_drop)
    X_train = X_train_full[common_cols]
    y_train_full = train_df[cfg.target_col]
    X_test = test_df[common_cols]
    
    X_tr, y_tr, X_val, y_val = split_validation_data(cfg, train_df, X_train, y_train_full)
    
    model, params = train_lightgbm(cfg, X_tr, y_tr, X_val, y_val)
    evaluate_and_save(cfg, model, params, X_val, y_val)
    create_submission(cfg, model, test_df, X_test)

if __name__ == "__main__":
    main()