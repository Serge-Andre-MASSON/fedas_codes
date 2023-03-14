import pandas as pd

from etl.extract.text_decoder_data import (
    split,
    add_special_tokens,
)


train_data_path = "data/train_technical_test.csv"
df = pd.read_csv(train_data_path)


def test_splits():
    fedas_code = "125465"
    splitted_fedas_code = ["1", "25", "46", "5"]
    assert split(fedas_code) == splitted_fedas_code


def test_add_special_tokens():
    splitted_fedas_code = ["1", "25", "46", "5"]
    assert add_special_tokens(splitted_fedas_code) == [
        "<sos>", "1", "25", "46", "5", "<eos>"
    ]


# def test_extract_dummy_text_decoder_data():
#     assert extract_dummy_text_decoder_data(5) == [
#         ["<sos>"],
#         ["<sos>"],
#         ["<sos>"],
#         ["<sos>"],
#         ["<sos>"],
#     ]
