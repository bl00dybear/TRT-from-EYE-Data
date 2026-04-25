from dataclasses import dataclass

@dataclass
class ModelConfig:
    train_path: str = ""
    test_path: str = ""
    output_dir: str = "output"
    model_filename: str = "model.txt"
    submission_filename: str = "submission.csv"
    validation_report_filename: str = "validation_report.txt"

    target_col: str = "answer"
    participant_col: str = "participant_id"
    id_col: str = "word_id"
    submission_id_col: str = "datapointID"

    categorical_features: tuple = ("NE_type", "POS_tag", "text_id")

    drop_cols: tuple = (
        "word_id", "word", "answer", "participant_id",
        "text", "text_id_raw"
    )

    num_leaves: int = 127
    learning_rate: float = 0.05
    n_estimators: int = 1000
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    random_state: int = 42
    early_stopping_rounds: int = 50
    verbose_eval: int = 100
    validation_participant_fraction: float = 0.2
