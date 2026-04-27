import os
import warnings

N_PHYSICAL_CORES = 32 


os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
os.environ["OMP_NUM_THREADS"] = str(N_PHYSICAL_CORES)
os.environ["MKL_NUM_THREADS"] = str(N_PHYSICAL_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_PHYSICAL_CORES)

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from scipy.stats import pearsonr

try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except ImportError:
    warnings.warn("scikit-learn-intelex nu este instalat. Recomandare: uv pip install scikit-learn-intelex")

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from config_model import BoostedTreesConfig


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

    cast_map_train = {}
    cast_map_test = {}

    for col in numeric_cols:
        cast_map_train[col] = np.float32
        cast_map_test[col] = np.float32

    for col in categorical_cols:
        cast_map_train[col] = np.int32
        cast_map_test[col] = np.int32

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
    model.fit(
        X_tr_cb,
        y_tr,
        eval_set=(X_val_cb, y_val),
        cat_features=cat_idx,
        use_best_model=True,
    )
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
        callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
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
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def main():
    cfg = BoostedTreesConfig()

    train_df = pd.read_csv(cfg.train_path)
    test_df = pd.read_csv(cfg.test_path)

    if cfg.target_col not in train_df.columns:
        raise ValueError(f"Missing target column: {cfg.target_col}")

    if cfg.group_col not in train_df.columns:
        raise ValueError(f"Missing group column: {cfg.group_col}")

    X_train, X_test, feature_cols, cat_idx = prepare_features(train_df, test_df, cfg)
    y = train_df[cfg.target_col].values.astype(np.float32)
    groups = train_df[cfg.group_col].astype(str).values

    unique_group_count = pd.Series(groups).nunique()
    n_splits = min(cfg.n_splits, unique_group_count)
    if n_splits < 2:
        raise ValueError("Need at least 2 unique participant groups for GroupKFold")

    gkf = GroupKFold(n_splits=n_splits)

    n_train = len(X_train)
    n_test = len(X_test)

    oof_cat = np.zeros(n_train, dtype=np.float32)
    oof_lgb = np.zeros(n_train, dtype=np.float32)
    oof_xgb = np.zeros(n_train, dtype=np.float32)

    test_cat_folds = []
    test_lgb_folds = []
    test_xgb_folds = []

    print(f"Using {len(feature_cols)} features")
    print(f"Running {n_splits} GroupKFold splits")

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y, groups), start=1):
        X_tr = X_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = y[tr_idx]
        y_val = y[val_idx]

        print(f"\nFold {fold}/{n_splits} | Train rows: {len(X_tr)} | Val rows: {len(X_val)}")

        cat_model = fit_catboost(X_tr, y_tr, X_val, y_val, cat_idx, cfg)
        lgb_model = fit_lightgbm(X_tr, y_tr, X_val, y_val, cfg)
        xgb_model = fit_xgboost(X_tr, y_tr, X_val, y_val, cfg)

        cat_val_pred = cat_model.predict(X_val).astype(np.float32)
        lgb_val_pred = lgb_model.predict(X_val).astype(np.float32)
        xgb_val_pred = xgb_model.predict(X_val).astype(np.float32)

        oof_cat[val_idx] = cat_val_pred
        oof_lgb[val_idx] = lgb_val_pred
        oof_xgb[val_idx] = xgb_val_pred

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

    cat_rmse, cat_r2, cat_pearson, cat_score = compute_metrics(y, oof_cat)
    lgb_rmse, lgb_r2, lgb_pearson, lgb_score = compute_metrics(y, oof_lgb)
    xgb_rmse, xgb_r2, xgb_pearson, xgb_score = compute_metrics(y, oof_xgb)

    oof_stack = np.column_stack([oof_cat, oof_lgb, oof_xgb])
    test_stack = np.column_stack([test_cat, test_lgb, test_xgb])

    meta_model = Ridge(alpha=cfg.meta_alpha)
    meta_model.fit(oof_stack, y)

    oof_meta = meta_model.predict(oof_stack).astype(np.float32)
    test_meta = meta_model.predict(test_stack).astype(np.float32)
    test_meta = np.clip(test_meta, 0, None)

    meta_rmse, meta_r2, meta_pearson, meta_score = compute_metrics(y, oof_meta)

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

    os.makedirs(cfg.output_dir, exist_ok=True)

    report_path = os.path.join(cfg.output_dir, cfg.model_report_filename)
    with open(report_path, "w") as f:
        f.write(f"Features used: {len(feature_cols)}\n")
        f.write(f"Folds: {n_splits}\n\n")

        f.write("CatBoost OOF\n")
        f.write(f"RMSE: {cat_rmse:.6f}\n")
        f.write(f"R2: {cat_r2:.6f}\n")
        f.write(f"Pearson: {cat_pearson:.6f}\n")
        f.write(f"Hackathon Score: {cat_score:.6f}\n\n")

        f.write("LightGBM OOF\n")
        f.write(f"RMSE: {lgb_rmse:.6f}\n")
        f.write(f"R2: {lgb_r2:.6f}\n")
        f.write(f"Pearson: {lgb_pearson:.6f}\n")
        f.write(f"Hackathon Score: {lgb_score:.6f}\n\n")

        f.write("XGBoost OOF\n")
        f.write(f"RMSE: {xgb_rmse:.6f}\n")
        f.write(f"R2: {xgb_r2:.6f}\n")
        f.write(f"Pearson: {xgb_pearson:.6f}\n")
        f.write(f"Hackathon Score: {xgb_score:.6f}\n\n")

        f.write("Stacked Ridge OOF\n")
        f.write(f"RMSE: {meta_rmse:.6f}\n")
        f.write(f"R2: {meta_r2:.6f}\n")
        f.write(f"Pearson: {meta_pearson:.6f}\n")
        f.write(f"Hackathon Score: {meta_score:.6f}\n")

    submission_path = os.path.join(cfg.output_dir, cfg.submission_filename)
    submission_df.to_csv(submission_path, index=False)

    print("\n=== OOF METRICS ===")
    print(f"CatBoost score: {cat_score:.4f}")
    print(f"LightGBM score: {lgb_score:.4f}")
    print(f"XGBoost score: {xgb_score:.4f}")
    print(f"Stacked score: {meta_score:.4f}")

    print("\n=== OUTPUT ===")
    print(f"Report: {report_path}")
    print(f"Submission: {submission_path}")
    print(f"Rows: {len(submission_df)}")


if __name__ == "__main__":
    main()