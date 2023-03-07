from typing import Iterable
import pandas as pd
from pandas import DataFrame, Series


hints_cols = [
    "accurate_gender",
    "article_main_category",
    "article_type",
    "article_detail",
    "comment",
]


def extract_text_encoder_data(df: DataFrame, max_length: int):
    hints = get_hints(df)
    brands = df["brand"].str.split()  # "brand752" -> ["brand752"]

    X_ = brands + hints
    return right_padded(X_, max_length)


def get_hints(df: DataFrame) -> Series:
    return df[hints_cols].fillna("")\
                         .agg(" ".join, axis=1)\
                         .str.lower()\
                         .replace(r"[-_%,;&./0-9()]", " ", regex=True)\
                         .replace(r" \w ", " ", regex=True)\
                         .replace(r" {2,}", " ", regex=True)\
                         .str.strip()\
                         .str.split()


def right_padded(X_: Iterable[list] | pd.Series, max_length: int = None):
    if max_length is None:
        max_length = X_.apply(len).max()
    for l in X_:
        while len(l) < max_length:
            l.append("<pad>")
    return X_
