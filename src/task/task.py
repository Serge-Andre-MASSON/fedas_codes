import pickle
from etl.etl import ETL, TrainData
from os import mkdir

from model.model import get_model
from trainer.trainer import Trainer


class Task:
    """Create a Task"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.etl = None

    def set_etl(self, etl: ETL) -> None:
        self.etl = etl

    def run(self):
        pass


class Training(Task):
    """Create a training Task without validation split. 
    The resulting model may be used for inference on unknown data.
    """

    def __init__(self, data_path):
        super().__init__(data_path)
        self.model_checkpoint = {
            "task": "training"
        }

    def run(self):
        etl = self.get_etl()
        etl.process()
        self.model_checkpoint["train_etl"] = etl
        len_encoder_vocab = etl.len_encoder_vocab
        len_decoder_vocab = etl.len_decoder_vocab

        model = get_model(len_encoder_vocab, len_decoder_vocab)
        train_dl = etl.dataloader

        trainer = Trainer(model)
        trainer.fit(train_dl, epochs=1)

        self.model_checkpoint["model_state_dict"] = model.state_dict()
        self.id = id(self.model_checkpoint)
        self.create_models_dir()
        with open(f"models/{self.id}.PICKLE", "wb") as f:
            pickle.dump(self.model_checkpoint, f)

    def get_etl(self) -> TrainData:
        if self.etl is None:
            etl = TrainData(self.data_path)
            self.set_etl(etl)
        return self.etl

    def create_models_dir(self):
        try:
            mkdir("models")
        except FileExistsError:
            pass


class ValidationTraining(Task):
    """Create a training Task with validation split. 
    This should be used for experiments.
    """

    def __init__(self, data_path, validation_split):
        super().__init__(data_path)
        self.validation_split = validation_split

    def run(self):
        print("Experimental training")


class Inference(Task):
    """Create a Task for inference with a pretrained model."""

    def __init__(self, data_path, model_checkpoint_path):
        super().__init__(data_path)
        self.model_checkpoint_path = model_checkpoint_path

    def run(self):
        print("Inference")
