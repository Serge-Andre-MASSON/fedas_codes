import pandas as pd

from etl.extract.text_decoder_data import (
    extract_text_decoder_data,
    extract_dummy_text_decoder_data
)
from etl.extract.text_encoder_data import extract_text_encoder_data


def extract_text_data(
    path: str,
    text_decoder_data: bool = True
):
    df = pd.read_csv(path)

    if text_decoder_data:
        return (extract_text_encoder_data(df),
                extract_text_decoder_data(df))
    else:
        encoder_data = extract_text_encoder_data(df)
        decoder_data = extract_dummy_text_decoder_data(len(encoder_data))
        return encoder_data, decoder_data
