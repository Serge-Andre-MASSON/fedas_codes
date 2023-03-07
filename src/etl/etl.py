import pickle
from etl.extract.text_data import extract_text_data
from etl.transform.tokenizer import Tokenizer, get_tokenizer
from etl.load.dataloader import get_dataloader


class ETL:
    def __init__(self, data_path):
        self.data_path = data_path


class TrainData(ETL):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.encoder_tokenizer = None
        self.decoder_tokenizer = None
        self.dataloader = None

    def process(self):
        text_encoder_data, text_decoder_data = extract_text_data(
            self.data_path)

        encoder_tokenizer = get_tokenizer(
            text_encoder_data,
            special_tokens=["<pad>"]
        )
        decoder_tokenizer = get_tokenizer(
            text_decoder_data,
            special_tokens=["<sos>", "<eos>"]
        )

        X = encoder_tokenizer.transform(text_encoder_data)
        self.encoder_tokenizer = encoder_tokenizer

        y = decoder_tokenizer.transform(text_decoder_data)
        self.decoder_tokenizer = decoder_tokenizer

        self.dataloader = get_dataloader(X, y)

    def get_encoder_tokenizer(self):
        return self.encoder_tokenizer

    def get_decoder_tokenizer(self):
        return self.decoder_tokenizer

    def get_dataloader(self):
        return self.dataloader


class InferenceData(ETL):
    def __init__(self, data_path, model_checkpoint):
        super().__init__(data_path)
        self.model_checkpoint = model_checkpoint

        self.dataloader = None

    def process(self):
        encoder_tokenizer: Tokenizer = self.model_checkpoint["encoder_tokenizer"]
        decoder_tokenizer: Tokenizer = self.model_checkpoint["decoder_tokenizer"]

        text_encoder_data, text_decoder_data = extract_text_data(
            self.data_path,
            encoder_max_length=20,
            text_decoder_data=False
        )

        X = encoder_tokenizer.transform(text_encoder_data)
        y = decoder_tokenizer.transform(text_decoder_data)

        self.dataloader = get_dataloader(X, y)

    def get_dataloader(self):
        return self.dataloader
