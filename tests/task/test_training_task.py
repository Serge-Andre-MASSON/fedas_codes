from os import rmdir

import pytest

from etl.etl import TrainData
from task.task import Training


@pytest.fixture
def task() -> Training:
    data_path = "data/dummy_train_technical_test.csv"
    task = Training(data_path)
    return task


def test_task_can_get_an_etl(task: Training):
    assert task.get_etl()


def test_task_write_etl_to_model_checkpoint(task: Training):
    task.run()
    checkpoint = task.model_checkpoint
    assert isinstance(checkpoint["train_etl"], TrainData)


def test_task_can_create_a_models_dir(task: Training):
    task.create_models_dir()
    rmdir("models")


def test_task_can_create_a_model_checkpoint(task: Training):
    checkpoint = task.model_checkpoint
    assert isinstance(checkpoint, dict)
    assert checkpoint["task"] == "training"
