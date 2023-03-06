from os import rmdir, path

import pytest

from etl.etl import TrainData
from etl.transform.tokenizer import Tokenizer
from task.task import Task, Training


@pytest.fixture
def task() -> Training:
    data_path = "data/dummy_train_technical_test.csv"
    task = Training(data_path)
    return task


@pytest.fixture
def runned_task(task: Task):
    task.run()
    return task


def test_task_write_etl_to_model_checkpoint(runned_task: Training):
    checkpoint = runned_task.model_checkpoint

    assert checkpoint["task"] == "training"

    assert isinstance(checkpoint, dict)
    assert isinstance(checkpoint["encoder_tokenizer"], Tokenizer)
    assert isinstance(checkpoint["decoder_tokenizer"], Tokenizer)
    assert isinstance(checkpoint["len_encoder_vocab"], int)
    assert isinstance(checkpoint["len_decoder_vocab"], int)


def test_task_can_create_a_models_dir(task: Training):
    task.create_models_dir()
    assert path.exists('models')
    try:
        rmdir("models")
    except OSError:
        pass
