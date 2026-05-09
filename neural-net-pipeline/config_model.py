from dataclasses import dataclass


@dataclass
class ModelConfig:
    train_path: str = "feature-pipeline/output/train_features.csv"
    test_path: str = "feature-pipeline/output/test_features.csv"
    output_dir: str = "neural-net-pipeline/output"
    model_filename: str = "best_model.pth"
    submission_filename: str = "submission.csv"
    validation_report_filename: str = "validation_report.txt"

    target_col: str = "answer"
    id_col: str = "word_id"
    participant_col: str = "participant_id"
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
        "page_num",
        "word_idx",
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
        "answer",
        "text",
        "text_id",
        "datapointID",
    )

    hidden_dims: tuple = (512, 256, 128)
    dropout: float = 0.25
    batch_size: int = 1024
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    epochs: int = 120
    estop_limit: int = 20
    scheduler_patience: int = 6
    scheduler_factor: float = 0.5
    val_fraction_participants: float = 0.2
    random_state: int = 42
    num_workers: int = 4

# === VALIDATION METRICS ===
# RMSE: 233.0662
# R2: 0.3015
# Pearson: 0.5845
# Hackathon Score: 44.2989

