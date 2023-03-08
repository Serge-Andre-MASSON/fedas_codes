from task.task import Task, Inference, Training, ValidationTraining


def get_task(
    train_data_path: str = None,
    validation_split: float = None,
    infer_data_path: str = None,
    model_checkpoint_path: str = None,
    model_name: str = None
) -> Task:
    if train_data_path is None:
        if infer_data_path is None or model_checkpoint_path is None:
            pass
        else:
            return Inference(infer_data_path, model_checkpoint_path)
    else:
        if validation_split is None:
            return Training(train_data_path, model_name)
        else:
            return ValidationTraining(train_data_path, validation_split)
