from pandas import DataFrame, Series


def extract_text_decoder_data(df: DataFrame) -> Series:
    """Extract correct fedas codes from source DataFrame and return
    them as a pd.Series in which each element is a tokenizer-ready version
    of the fedas code e.g. : ["<sos>", "1", "25", "46", "5", "<eos>"]

    Returns:
        _type_: pd.Series
    """
    return df["correct_fedas_code"].astype(str)\
                                   .apply(split)\
                                   .apply(add_special_tokens)


def split(c: str) -> list[str]:
    return [c[0], c[1:3], c[3:5], c[-1]]


def add_special_tokens(l: list[str]) -> list[str]:
    return ["<sos>"] + l + ["<eos>"]
