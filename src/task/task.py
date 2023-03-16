import numpy as np
from pathlib import Path
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from etl.extract.text_data import extract_text_data
import pickle
from tqdm import tqdm
from etl.load.dataloader import get_dataloader
from etl.transform.tokenizer import Tokenizer

from model.model import get_model
from trainer.trainer import Trainer
from trainer.summary import TrainingSummary

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


class Task:
    """Create a Task"""

    def __init__(self, data_path):
        self.data_path = Path(data_path)

    def run(self):
        pass


class Training(Task):
    """Create a training Task without validation split. 
    The resulting model may be used for inference on unknown data.
    """

    def __init__(self, data_path, model_name: str = None) -> None:
        super().__init__(data_path)

        self.model_name = model_name or id(self)
        self.checkpoint_path = MODEL_DIR / f"{self.model_name}.PICKLE"

        self.encoder_tokenizer = Tokenizer(
            special_tokens=["<pad>"],
            min_freq=3
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

        summary = TrainingSummary()
        trainer = Trainer(model, summary)

        try:
            trainer.fit(dl)
        except KeyboardInterrupt:
            self.checkpoint["model"] = model
            self.checkpoint["training_summary"] = summary

            self.save()
            sys.exit()

        self.checkpoint["model"] = model
        self.checkpoint["training_summary"] = summary

        if save:
            self.save()

    def save(self):
        with open(self.checkpoint_path, "wb") as f:
            pickle.dump(self.checkpoint, f)

        print(self.checkpoint_path)


class ValidationTraining(Task):
    """Create a training Task with validation split. 
    This should be used for experiments.
    """

    def __init__(self, data_path, validation_split, model_name: str = None, random_state: int = 42):
        super().__init__(data_path)
        self.validation_split = validation_split
        self.random_state = random_state

        model_name = model_name or id(self)

        self.encoder_tokenizer = Tokenizer(
            special_tokens=["<pad>"],
            min_freq=3
        )
        self.decoder_tokenizer = Tokenizer(
            special_tokens=["<sos>", "<eos>"]
        )

        self.checkpoint_path = MODEL_DIR / f"{model_name}_val.PICKLE"

        self.checkpoint = {
            "model_name": model_name
        }

    def run(self):
        text_encoder_data, text_decoder_data = extract_text_data(
            self.data_path)

        X_train, X_test, y_train, y_test = train_test_split(
            text_encoder_data,
            text_decoder_data,
            test_size=self.validation_split,
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

        summary = TrainingSummary()
        trainer = Trainer(model, summary)

        try:
            trainer.fit(train_dl, test_dl)
        except KeyboardInterrupt:
            self.checkpoint["model"] = model
            self.checkpoint["training_summary"] = summary
            self.save()
            sys.exit()

        self.checkpoint["model"] = model
        self.checkpoint["training_summary"] = summary
        self.save()

    def save(self):
        with open(self.checkpoint_path, "wb") as f:
            pickle.dump(self.checkpoint, f)

        print(self.checkpoint_path)


class Inference(Task):
    """Create a Task for inference with a pretrained model."""

    def __init__(self, data_path, model_name):
        super().__init__(data_path)
        self.model_name = model_name
        self.checkpoint_path = MODEL_DIR / f"{self.model_name}.PICKLE"

    def run(self):
        with open(self.checkpoint_path, "rb") as f:
            self.checkpoint = pickle.load(f)

        encoder_tokenizer: Tokenizer = self.checkpoint["encoder_tokenizer"]
        decoder_tokenizer: Tokenizer = self.checkpoint["decoder_tokenizer"]

        model = self.checkpoint["model"]

        text_encoder_data = extract_text_data(
            self.data_path,
            inference=True
        )

        encoder_data = encoder_tokenizer.transform(text_encoder_data)
        dl = get_dataloader(encoder_data)

        raw_pred = None
        raw_prob = None

        for batch in tqdm(dl):
            prob, pred = model.predict(batch)
            raw_pred = pred if raw_pred is None else torch.cat(
                [raw_pred, pred])
            raw_prob = prob if raw_prob is None else torch.cat(
                [raw_prob, prob])

        output_df = pd.read_csv(self.data_path)

        raw_pred = raw_pred.cpu().numpy()
        raw_prob = raw_prob.cpu().numpy()

        output_df["predicted_fedas_code"] = decoder_tokenizer.inverse_transform(
            raw_pred
        )

        output_df["predicted_fedas_code"] = output_df["predicted_fedas_code"].apply(
            lambda l: (''.join(c for c in l))
        ).apply(to_int)

        prob_df = pd.DataFrame(raw_prob)
        output_df = pd.concat([output_df, prob_df], axis=1)

        output_path = Path(f"models/{self.model_name}.csv")
        output_df.to_csv(output_path, index=False)

        return output_df


def to_int(s):
    try:
        return int(s)
    except:
        s = s[:-1]
        return to_int(s)
