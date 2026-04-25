from dataclasses import dataclass

@dataclass
class MorphConfig:
    output_dir: str = "output"
    reconstructed_pages_filename: str = "reconstructed_pages.csv"
    morph_output_filename: str = "morph_features.csv"
    spacy_model: str = "ro_core_news_lg"
    batch_size: int = 32
    gpu_id: int = 0
    word_id_sequence_column: str = "word_id_sequence"
    word_sequence_column: str = "word_sequence"
    reconstructed_text_column: str = "reconstructed_text"
    case_order: tuple = ("Nom", "Acc", "Dat", "Gen", "Voc")
    verb_form_order: tuple = ("Fin", "Inf", "Part", "Ger", "Sup")
