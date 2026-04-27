from dataclasses import dataclass


@dataclass
class BoostedTreesConfig:
    train_path: str = "merged_output/train_final.csv"
    test_path: str = "merged_output/test_final.csv"
    output_dir: str = "train-boosted-trees/output"
    model_report_filename: str = "stacking_report.txt"
    submission_filename: str = "submission.csv"

    target_col: str = "answer"
    group_col: str = "participant_id"
    id_col: str = "word_id"
    submission_id_col: str = "datapointID"

    drop_cols: tuple = (
        "word_id",
        "word",
        "answer",
        "text_id",
    )

    categorical_features: tuple = (
        "text",
        "dep_rel",
        "NE_type",
        "POS_tag",
        "verb_form",
        "case_encoded",
    )

    n_splits: int = 5
    random_state: int = 42
    n_jobs: int = 40

    cat_iterations: int = 6000
    cat_depth: int = 7
    cat_learning_rate: float = 0.03

    lgb_num_leaves: int = 127
    lgb_learning_rate: float = 0.03
    lgb_n_estimators: int = 6000
    lgb_min_child_samples: int = 20
    lgb_subsample: float = 0.85
    lgb_colsample_bytree: float = 0.85
    lgb_reg_alpha: float = 0.1
    lgb_reg_lambda: float = 1.0

    xgb_max_depth: int = 7
    xgb_learning_rate: float = 0.03
    xgb_n_estimators: int = 6000
    xgb_subsample: float = 0.85
    xgb_colsample_bytree: float = 0.85
    xgb_reg_alpha: float = 0.1
    xgb_reg_lambda: float = 1.0

    early_stopping_rounds: int = 200
    verbose_eval: int = 200
    meta_alpha: float = 1.0