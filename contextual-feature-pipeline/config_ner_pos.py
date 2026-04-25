from dataclasses import dataclass

@dataclass
class NerPosConfig:
    output_dir: str = "output"
    reconstructed_pages_filename: str = "reconstructed_pages.csv"
    ner_pos_output_filename: str = "ner_pos_features.csv"
    spacy_model: str = "ro_core_news_lg"
    batch_size: int = 32
    gpu_id: int = 0
    word_id_sequence_column: str = "word_id_sequence"
    word_sequence_column: str = "word_sequence"
    reconstructed_text_column: str = "reconstructed_text"
    content_pos_tags: tuple = ("NOUN", "VERB", "ADJ", "ADV", "PROPN")
