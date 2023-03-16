from pathlib import Path
import pandas as pd

from etl.extract.text_encoder_data import extract_text_encoder_data
from etl.extract.text_decoder_data import extract_text_decoder_data


def extract_text_data(
    path: Path,
    inference=False
):
    df = pd.read_csv(path)

    if inference:
        return extract_text_encoder_data(df)
    else:
        return (extract_text_encoder_data(df),
                extract_text_decoder_data(df))
