from os import mkdir, rmdir
import pytest
from etl.etl import ETL
from task.task import Task


@pytest.fixture
def task():
    data_path = "path/to/some/data"
    task = Task(data_path)
    return task


def test_task_has_a_data_path(task: Task):
    assert task.data_path
