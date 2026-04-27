import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from config_model import BoostedTreesConfig

N_PHYSICAL_CORES = 32 

os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
os.environ["OMP_NUM_THREADS"] = str(N_PHYSICAL_CORES)
os.environ["MKL_NUM_THREADS"] = str(N_PHYSICAL_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_PHYSICAL_CORES)

try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except ImportError:
    pass

def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = max(0.0, r2_score(y_true, y_pred))

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson = 0.0
    else:
        p = pearsonr(y_true, y_pred)[0]
        pearson = float(abs(p)) if not np.isnan(p) else 0.0

    score = 100.0 * (r2 + pearson) / 2.0
    return rmse, r2, pearson, score

def sort_dataframes(train_df, test_df):
    sort_cols = [c for c in ['participant_id', 'text', 'page_num', 'word_idx'] if c in train_df.columns]
    if len(sort_cols) > 0:
        train_df = train_df.sort_values(by=sort_cols)
        test_df = test_df.sort_values(by=sort_cols)
    return train_df, test_df

def add_lag_lead_columns(train_df, test_df, cols_to_shift):
    group_cols = [c for c in ['participant_id', 'text'] if c in train_df.columns]
    if len(group_cols) == 0 and 'participant_id' in train_df.columns:
        group_cols = ['participant_id']

    for col in cols_to_shift:
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
            
    return train_df, test_df

def prepare_features(train_df, test_df, cfg):
    train_df = train_df.copy()
    test_df = test_df.copy()

    for col in cfg.categorical_features:
        if col not in train_df.columns:
            train_df[col] = "__NA__"
        if col not in test_df.columns:
            test_df[col] = "__NA__"

    cols_to_drop = set(cfg.drop_cols)
    cols_to_drop.add(cfg.target_col)
    cols_to_drop.add(cfg.group_col)

    feature_cols = [
        c for c in train_df.columns
        if c in test_df.columns and c not in cols_to_drop
    ]

    numeric_cols = [
        c for c in feature_cols
        if c not in cfg.categorical_features and pd.api.types.is_numeric_dtype(train_df[c])
    ]
    categorical_cols = [c for c in cfg.categorical_features if c in feature_cols]

    for col in numeric_cols:
        med = pd.to_numeric(train_df[col], errors="coerce").median()
        if np.isnan(med):
            med = 0.0
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce").fillna(med)
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce").fillna(med)

    for col in categorical_cols:
        tr = train_df[col].astype(str).fillna("__NA__")
        te = test_df[col].astype(str).fillna("__NA__")
        uniq = pd.Index(pd.concat([tr, te], ignore_index=True).unique())
        mapping = {v: i for i, v in enumerate(uniq)}

        train_df[col] = tr.map(mapping).astype(np.int32)
        test_df[col] = te.map(mapping).astype(np.int32)

    ordered_cols = numeric_cols + categorical_cols
    X_train = train_df[ordered_cols].copy()
    X_test = test_df[ordered_cols].copy()

    cast_map_train = {col: np.float32 for col in numeric_cols}
    cast_map_test = {col: np.float32 for col in numeric_cols}
    
    cast_map_train.update({col: np.int32 for col in categorical_cols})
    cast_map_test.update({col: np.int32 for col in categorical_cols})

    if cast_map_train:
        X_train = X_train.astype(cast_map_train)
        X_test = X_test.astype(cast_map_test)

    cat_idx = [ordered_cols.index(c) for c in categorical_cols]
    return X_train, X_test, ordered_cols, cat_idx

def fit_catboost(X_tr, y_tr, X_val, y_val, cat_idx, cfg):
    X_tr_cb = X_tr.copy()
    X_val_cb = X_val.copy()

    if cat_idx:
        cat_cols = [X_tr_cb.columns[i] for i in cat_idx]
        X_tr_cb = X_tr_cb.astype({col: str for col in cat_cols})
        X_val_cb = X_val_cb.astype({col: str for col in cat_cols})

    model = CatBoostRegressor(
        iterations=cfg.cat_iterations,
        depth=cfg.cat_depth,
        learning_rate=cfg.cat_learning_rate,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=cfg.random_state,
        thread_count=N_PHYSICAL_CORES,
        od_type="Iter",
        od_wait=cfg.early_stopping_rounds,
        verbose=False,
    )
    model.fit(X_tr_cb, y_tr, eval_set=(X_val_cb, y_val), cat_features=cat_idx, use_best_model=True)
    return model

def fit_lightgbm(X_tr, y_tr, X_val, y_val, cfg):
    model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        num_leaves=cfg.lgb_num_leaves,
        learning_rate=cfg.lgb_learning_rate,
        n_estimators=cfg.lgb_n_estimators,
        min_child_samples=cfg.lgb_min_child_samples,
        subsample=cfg.lgb_subsample,
        colsample_bytree=cfg.lgb_colsample_bytree,
        reg_alpha=cfg.lgb_reg_alpha,
        reg_lambda=cfg.lgb_reg_lambda,
        random_state=cfg.random_state,
        n_jobs=N_PHYSICAL_CORES,
        verbose=-1,
    )
    model.fit(
        X_tr, 
        y_tr, 
        eval_set=[(X_val, y_val)], 
        eval_metric="rmse", 
        callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)]
    )
    return model

def fit_xgboost(X_tr, y_tr, X_val, y_val, cfg):
    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        max_depth=cfg.xgb_max_depth,
        learning_rate=cfg.xgb_learning_rate,
        n_estimators=cfg.xgb_n_estimators,
        subsample=cfg.xgb_subsample,
        colsample_bytree=cfg.xgb_colsample_bytree,
        reg_alpha=cfg.xgb_reg_alpha,
        reg_lambda=cfg.xgb_reg_lambda,
        random_state=cfg.random_state,
        n_jobs=N_PHYSICAL_CORES,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model

def run_cross_validation(X_train, y, groups, X_test, cat_idx, cfg):
    unique_group_count = pd.Series(groups).nunique()
    n_splits = min(cfg.n_splits, unique_group_count)
    if n_splits < 2:
        raise ValueError("Need at least 2 unique participant groups for GroupKFold")

    gkf = GroupKFold(n_splits=n_splits)
    
    n_train = len(X_train)
    oof_cat = np.zeros(n_train, dtype=np.float32)
    oof_lgb = np.zeros(n_train, dtype=np.float32)
    oof_xgb = np.zeros(n_train, dtype=np.float32)

    test_cat_folds, test_lgb_folds, test_xgb_folds = [], [], []

    for tr_idx, val_idx in gkf.split(X_train, y, groups):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        cat_model = fit_catboost(X_tr, y_tr, X_val, y_val, cat_idx, cfg)
        lgb_model = fit_lightgbm(X_tr, y_tr, X_val, y_val, cfg)
        xgb_model = fit_xgboost(X_tr, y_tr, X_val, y_val, cfg)

        oof_cat[val_idx] = cat_model.predict(X_val).astype(np.float32)
        oof_lgb[val_idx] = lgb_model.predict(X_val).astype(np.float32)
        oof_xgb[val_idx] = xgb_model.predict(X_val).astype(np.float32)

        X_test_cb = X_test.copy()
        if cat_idx:
            cat_cols = [X_test_cb.columns[i] for i in cat_idx]
            X_test_cb = X_test_cb.astype({col: str for col in cat_cols})

        test_cat_folds.append(cat_model.predict(X_test_cb).astype(np.float32))
        test_lgb_folds.append(lgb_model.predict(X_test).astype(np.float32))
        test_xgb_folds.append(xgb_model.predict(X_test).astype(np.float32))

    test_cat = np.mean(np.vstack(test_cat_folds), axis=0)
    test_lgb = np.mean(np.vstack(test_lgb_folds), axis=0)
    test_xgb = np.mean(np.vstack(test_xgb_folds), axis=0)

    return oof_cat, oof_lgb, oof_xgb, test_cat, test_lgb, test_xgb, n_splits

def train_meta_model(y, oof_cat, oof_lgb, oof_xgb, test_cat, test_lgb, test_xgb, cfg):
    oof_stack = np.column_stack([oof_cat, oof_lgb, oof_xgb])
    test_stack = np.column_stack([test_cat, test_lgb, test_xgb])

    meta_model = Ridge(alpha=cfg.meta_alpha)
    meta_model.fit(oof_stack, y)

    oof_meta = meta_model.predict(oof_stack).astype(np.float32)
    test_meta = meta_model.predict(test_stack).astype(np.float32)
    test_meta = np.clip(test_meta, 0, None)

    return oof_meta, test_meta

def save_report(cfg, y, oof_cat, oof_lgb, oof_xgb, oof_meta, feature_cols, n_splits):
    cat_metrics = compute_metrics(y, oof_cat)
    lgb_metrics = compute_metrics(y, oof_lgb)
    xgb_metrics = compute_metrics(y, oof_xgb)
    meta_metrics = compute_metrics(y, oof_meta)

    os.makedirs(cfg.output_dir, exist_ok=True)
    report_path = os.path.join(cfg.output_dir, cfg.model_report_filename)
    
    with open(report_path, "w") as f:
        f.write(f"Features used: {len(feature_cols)}\n")
        f.write(f"Folds: {n_splits}\n\n")
        
        models_data = [
            ("CatBoost", cat_metrics),
            ("LightGBM", lgb_metrics),
            ("XGBoost", xgb_metrics),
            ("Stacked Ridge", meta_metrics)
        ]
        
        for name, metrics in models_data:
            f.write(f"{name} OOF\n")
            f.write(f"RMSE: {metrics[0]:.6f}\n")
            f.write(f"R2: {metrics[1]:.6f}\n")
            f.write(f"Pearson: {metrics[2]:.6f}\n")
            f.write(f"Hackathon Score: {metrics[3]:.6f}\n\n")
            
    print(f"Report saved to: {report_path}")

def create_submission(cfg, test_df, test_meta):
    n_test = len(test_df)
    if cfg.submission_id_col in test_df.columns:
        datapoint_ids = test_df[cfg.submission_id_col].values
    else:
        datapoint_ids = np.arange(n_test)

    submission_df = pd.DataFrame(
        {
            "subtaskID": np.ones(n_test, dtype=int),
            "datapointID": datapoint_ids,
            "answer": test_meta,
        }
    )

    submission_df["datapointID"] = pd.to_numeric(submission_df["datapointID"], errors="coerce")
    submission_df = submission_df.sort_values("datapointID").reset_index(drop=True)

    if submission_df["datapointID"].duplicated().any():
        submission_df = (
            submission_df.groupby("datapointID", as_index=False)
            .agg({"subtaskID": "first", "answer": "mean"})
            .sort_values("datapointID")
            .reset_index(drop=True)
        )

    submission_path = os.path.join(cfg.output_dir, cfg.submission_filename)
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to: {submission_path}")

def main():
    cfg = BoostedTreesConfig()

    train_df = pd.read_csv(cfg.train_path)
    test_df = pd.read_csv(cfg.test_path)

    train_df, test_df = sort_dataframes(train_df, test_df)
    
    cols_to_shift = ['word_len', 'contextual_surprisal', 'zipf_frequency']
    train_df, test_df = add_lag_lead_columns(train_df, test_df, cols_to_shift)

    X_train, X_test, feature_cols, cat_idx = prepare_features(train_df, test_df, cfg)
    y = train_df[cfg.target_col].values.astype(np.float32)
    groups = train_df[cfg.group_col].astype(str).values

    oof_cat, oof_lgb, oof_xgb, test_cat, test_lgb, test_xgb, n_splits = run_cross_validation(
        X_train, y, groups, X_test, cat_idx, cfg
    )

    oof_meta, test_meta = train_meta_model(
        y, oof_cat, oof_lgb, oof_xgb, test_cat, test_lgb, test_xgb, cfg
    )

    save_report(cfg, y, oof_cat, oof_lgb, oof_xgb, oof_meta, feature_cols, n_splits)
    create_submission(cfg, test_df, test_meta)

if __name__ == "__main__":
    main()