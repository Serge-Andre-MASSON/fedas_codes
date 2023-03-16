from typing import Iterable
import pandas as pd
from pandas import DataFrame, Series


hints_cols = [
    "accurate_gender",
    "article_main_category",
    "article_type",
    "article_detail",
    "comment",
    "model_label"
]


def extract_text_encoder_data(df: DataFrame):
    hints = get_hints(df)
    brands = df["brand"].str.split()  # "brand752" -> ["brand752"]

    X_ = brands + hints
    return right_padded(X_)


def get_hints(df: DataFrame) -> Series:
    return df[hints_cols].fillna("")\
                         .agg(" ".join, axis=1)\
                         .apply(drop_after_semi_col)\
                         .str.lower()\
                         .replace(r"[-_%,&./0-9()]", " ", regex=True)\
                         .replace(r" \w ", " ", regex=True)\
                         .replace(r" {2,}", " ", regex=True)\
                         .str.strip()\
                         .str.split()


def right_padded(X_: Iterable[list] | pd.Series):
    max_length = X_.apply(len).max()
    for i, l in enumerate(X_):
        while len(l) < max_length:
            l.append("<pad>")
    return X_


# Hack
def drop_after_semi_col(sequence: str):
    if ";" in sequence:
        semi_col_index = sequence.index(";")
        return sequence[:semi_col_index]
    return sequence
