from pathlib import Path
from task.task import Task, Inference, Training, ValidationTraining


def get_task(
    train_data_path: str = None,
    validation_split: float = None,
    test_data_path: str = None,
    model_name: str = None
) -> Task:
    """Return a task according to the provided kwargs."""
    if train_data_path is None:
        if test_data_path is None or model_name is None:
            pass
        else:
            return Inference(test_data_path, model_name)
    else:
        if validation_split is None:
            return Training(train_data_path, model_name)
        else:
            return ValidationTraining(train_data_path, validation_split, model_name)
