import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from etl.extract.text_data import extract_text_data
import pickle
from os import mkdir
from etl.load.dataloader import get_dataloader
from etl.transform.tokenizer import Tokenizer

from model.model import get_model
from trainer.trainer import Trainer


class Task:
    """Create a Task"""

    def __init__(self, data_path):
        self.data_path = data_path

    def run(self):
        pass


class Training(Task):
    """Create a training Task without validation split. 
    The resulting model may be used for inference on unknown data.
    """

    def __init__(self, data_path, model_name: str = None) -> None:
        super().__init__(data_path)
        self.model_name = model_name or id(self)
        self.encoder_tokenizer = Tokenizer(
            special_tokens=["<pad>"]
        )
        self.decoder_tokenizer = Tokenizer(
            special_tokens=["<sos>", "<eos>"]
        )
        self.checkpoint = {
            "model_name": self.model_name
        }

    def run(self, save: bool = True):
        text_encoder_data, text_decoder_data = extract_text_data(
            self.data_path)

        encoder_data = self.encoder_tokenizer.fit_transform(text_encoder_data)
        decoder_data = self.decoder_tokenizer.fit_transform(text_decoder_data)

        dl = get_dataloader(encoder_data, decoder_data)

        self.checkpoint["encoder_tokenizer"] = self.encoder_tokenizer
        self.checkpoint["decoder_tokenizer"] = self.decoder_tokenizer

        len_encoder_vocab = self.encoder_tokenizer.len_vocab

        len_decoder_vocab = self.decoder_tokenizer.len_vocab

        model = get_model(len_encoder_vocab, len_decoder_vocab)

        trainer = Trainer(model)
        trainer.fit(dl, epochs=150)

        self.checkpoint["model_state_dict"] = model.state_dict()
        self.checkpoint["model"] = model

        if save:
            checkpoint_path = f"models/{self.model_name}.PICKLE"
            self.create_models_dir()
            with open(checkpoint_path, "wb") as f:
                pickle.dump(self.checkpoint, f)
            print(checkpoint_path)

    def create_models_dir(self):
        try:
            mkdir("models")
        except FileExistsError:
            pass


class ValidationTraining(Task):
    """Create a training Task with validation split. 
    This should be used for experiments.
    """

    def __init__(self, data_path, validation_split, random_state: int = 42):
        super().__init__(data_path)
        self.validation_split = validation_split
        self.random_state = random_state

        self.encoder_tokenizer = Tokenizer(
            special_tokens=["<pad>"]
        )
        self.decoder_tokenizer = Tokenizer(
            special_tokens=["<sos>", "<eos>"]
        )

    def run(self):
        text_encoder_data, text_decoder_data = extract_text_data(
            self.data_path)

        X_train, X_test, y_train, y_test = train_test_split(
            text_encoder_data,
            text_decoder_data,
            test_size=self.validation_split,
            random_state=self.random_state
        )

        X_train = self.encoder_tokenizer.fit_transform(X_train)
        y_train = self.decoder_tokenizer.fit_transform(y_train)

        X_test = self.encoder_tokenizer.transform(X_test)
        y_test = self.decoder_tokenizer.transform(y_test)

        train_dl = get_dataloader(X_train, y_train)
        test_dl = get_dataloader(X_test, y_test)

        len_encoder_vocab = self.encoder_tokenizer.len_vocab
        len_decoder_vocab = self.decoder_tokenizer.len_vocab

        model = get_model(len_encoder_vocab, len_decoder_vocab)

        trainer = Trainer(model)
        trainer.fit(train_dl, test_dl, epochs=200)


class Inference(Task):
    """Create a Task for inference with a pretrained model."""

    def __init__(self, data_path, checkpoint_path):
        super().__init__(data_path)
        self.checkpoint_path = checkpoint_path

    def run(self):
        with open(self.checkpoint_path, "rb") as f:
            self.checkpoint = pickle.load(f)

        text_encoder_data, text_decoder_data = extract_text_data(
            self.data_path,
            text_decoder_data=False
        )

        encoder_tokenizer: Tokenizer = self.checkpoint["encoder_tokenizer"]
        decoder_tokenizer: Tokenizer = self.checkpoint["decoder_tokenizer"]

        encoder_data = encoder_tokenizer.transform(text_encoder_data)
        decoder_data = decoder_tokenizer.transform(text_decoder_data)

        dl = get_dataloader(
            encoder_data,
            decoder_data,
            shuffle=False,
            batch_size=2048
        )

        len_encoder_vocab = encoder_tokenizer.len_vocab
        len_decoder_vocab = decoder_tokenizer.len_vocab

        model = get_model(len_encoder_vocab, len_decoder_vocab)
        model_state_dict = self.checkpoint["model_state_dict"]

        model.load_state_dict(model_state_dict)
        raw_pred = None
        for batch in dl:
            X, _, _ = batch
            pred = model.predict(X)
            if raw_pred is None:
                raw_pred = pred
            else:
                raw_pred = torch.cat([raw_pred, pred])

        output_df = pd.read_csv(self.data_path)

        raw_pred = raw_pred.detach().cpu().numpy()
        output_df["predicted_fedas_code"] = decoder_tokenizer.inverse_transform(
            raw_pred
        )

        output_df["predicted_fedas_code"] = output_df["predicted_fedas_code"].apply(
            lambda l: int(''.join(c for c in l))
        )
        output_df.to_csv(f"{self.checkpoint_path}.csv", index=False)
