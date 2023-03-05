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

    def run(self):
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
