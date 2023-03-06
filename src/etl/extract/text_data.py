import pandas as pd

from etl.extract.text_decoder_data import extract_text_decoder_data
from etl.extract.text_encoder_data import extract_text_encoder_data


def extract_text_data(path: str):
    df = pd.read_csv(path)
    return (extract_text_encoder_data(df),
            extract_text_decoder_data(df))
