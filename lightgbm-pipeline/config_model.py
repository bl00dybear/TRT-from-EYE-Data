from dataclasses import dataclass

@dataclass
class ModelConfig:
    train_path: str = "feature-pipeline/output/train_features.csv"
    test_path: str = "feature-pipeline/output/test_features.csv"
    output_dir: str = "output"
    model_filename: str = "model.txt"
    submission_filename: str = "submission.csv"
    validation_report_filename: str = "validation_report.txt"

    target_col: str = "answer"
    participant_col: str = "participant_id"
    id_col: str = "word_id"
    submission_id_col: str = "datapointID"

    categorical_features: tuple = ()

    numeric_features: tuple = (
        "word_len",
        "word_len_sq",
        "n_diacritics",
        "is_capitalized",
        "starts_sentence",
        "ends_punct",
        "has_non_alpha",
        "zipf_frequency",
        "n_syllables",
        "vowel_consonant_ratio",
        "is_stopword",
        "participant_id",
        "is_NE",
        "NE_type",
        "POS_tag",
        "is_content_word",
        "contextual_surprisal",
        "morph_count",
        "case_encoded",
        "verb_form",
        "is_inflected",
    )

    drop_cols: tuple = (
        "word_id",
        "word",
        "word_idx",
        "answer",
        "text",
        "text_id",
        "datapointID",
    )
    
    categorical_cols = ['NE_type', 'POS_tag', 'verb_form', 'case_encoded']
    cols_to_shift = ['word_len', 'contextual_surprisal', 'zipf_frequency']

    num_leaves: int = 255
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
