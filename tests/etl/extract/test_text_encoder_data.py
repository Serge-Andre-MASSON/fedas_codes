import pandas as pd
from etl.extract.text_encoder_data import (
    get_text_encoder_data,
    get_hints,
    right_padded
)


train_data_path = "data/dummy_train_technical_test.csv"
df = pd.read_csv(train_data_path)

# Don't want to write tests for this.
# The way it works may be show in a
# dedicated DataFrame
