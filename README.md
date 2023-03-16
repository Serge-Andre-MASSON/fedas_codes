# Description

A transformer model train to predict fedas codes from a text description. Each fedas code is seen as 4 sub codes :

    123456 <--> 1 23 45 6

Sample of the data that should be feed to the model is present in example_data/.

# Installation
Clone the repo and run

```console
$ pip install -e .
```


# Usage
## CLI
You can run a full training task and save the trained model by running
```console
$ fedas train <training_data.csv> -n <modele_name>
```
For experiment with a validation split (say 0.3) you max run
```console
$ fedas train <training_data.csv> -v .3
```

Finally, for inference on new data with a trained model run
```console
$ fedas predict <unseen_data.csv> -n <model_name> 
```

## Notebook
You can instanciate Training, ValidationTraining, Inference classes and run them in a notebook, e.g. :
```python
from task.task import Training


data_path = "example_data/example_train_technical_test.csv"

task = Training(
    data_path=data_path,
    model_name="example_model"
)

task.run()  
```

# Remarks
This repo does not intend to provide a fancy state of the art model. The main objective I've been chasing is to implement a consistent workflow and a versatile codebase. 

As a result, with a few improvements (soon to come), it will be easy to improve the model while working in an almost production ready environment.