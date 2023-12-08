# ML XGboost regression for Slope reinforcement calculation


## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the dvc.yaml


# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/FBrownp/ML-geo
```
### STEP 01- Create a conda environment after opening the repository

```bash
python -m venv venv
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```


### DVC cmd

1. dvc init
2. dvc repro
3. dvc dag



## MLflow

[Documentation](https://mlflow.org/docs/latest/index.html)


##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/FBrownp/ML-geo.mlflow \
MLFLOW_TRACKING_USERNAME=FBrownp \
MLFLOW_TRACKING_PASSWORD=cceabdf6d4f5adc126b6cb03cc7cc4bf568e2591 \
python script.py
Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/FBrownp/ML-geo.mlflow

export MLFLOW_TRACKING_USERNAME=FBrownp

export MLFLOW_TRACKING_PASSWORD=cceabdf6d4f5adc126b6cb03cc7cc4bf568e2591

```
