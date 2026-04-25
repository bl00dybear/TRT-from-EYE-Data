from dataclasses import dataclass

@dataclass
class SurprisalConfig:
    output_dir: str = "output"
    reconstructed_pages_filename: str = "reconstructed_pages.csv"
    surprisal_output_filename: str = "surprisal_features.csv"
    model_name: str = "readerbench/ro-gpt2"
    gpu_id: int = 1
    max_tokens: int = 900
    cache_clear_every_n_pages: int = 100
    word_id_sequence_column: str = "word_id_sequence"
    word_sequence_column: str = "word_sequence"
    reconstructed_text_column: str = "reconstructed_text"
