from task import get_task, Inference, Training, ValidationTraining


def test_task_is_training():
    kwargs = {
        "train_data_path": "data/dummy_train_technical_test.csv",
        "validation_split": None,
    }
    task = get_task(**kwargs)
    assert isinstance(task, Training)


def test_task_is_validation_training():
    kwargs = {
        "train_data_path": "data/dummy_train_technical_test.csv",
        "validation_split": .2,
    }
    task = get_task(**kwargs)
    assert isinstance(task, ValidationTraining)


def test_task_is_inference():
    kwargs = {
        "infer_data_path": "data/dummy_test_technical_test.csv",
        "model_checkpoint_path": "path/to/some/model/checkpoint",
    }
    task = get_task(**kwargs)
    assert isinstance(task, Inference)
