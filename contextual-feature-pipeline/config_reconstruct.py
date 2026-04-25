from dataclasses import dataclass

@dataclass
class ReconstructConfig:
    train_path: str = "../data/train_data.csv"
    test_path: str = "../data/test_data.csv"
    output_dir: str = "output"
    reconstructed_pages_filename: str = "reconstructed_pages.csv"
    word_id_map_filename: str = "word_id_map.csv"
    word_id_column: str = "word_id"
    word_column: str = "word"
