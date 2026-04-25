from dataclasses import dataclass

@dataclass
class AggregateConfig:
    output_dir: str = "output"
    word_id_map_filename: str = "word_id_map.csv"
    ner_pos_filename: str = "ner_pos_features.csv"
    surprisal_filename: str = "surprisal_features.csv"
    morph_filename: str = "morph_features.csv"
    train_path: str = "../data/train_data.csv"
    test_path: str = "../data/test_data.csv"
    train_output_filename: str = "train_features.csv"
    test_output_filename: str = "test_features.csv"
