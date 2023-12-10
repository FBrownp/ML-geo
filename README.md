# A complete data science and machine learning project using CI/CD Pipelines


This project consists of an EDA, and machine learning model formulation to predict the optimized support system for slopes by calculating the equilibrium pressure by a machine learning model for different soil parameters and geometric properties. This project also have the complete CI/CD Pipeline for the machine learning model using MlFlow to track the version of the model and using DVC to ingest data, transform data, train the model, evaluate and predict. FastAPI is used to create a Restful API to interact with the model. GitHub Workflow is used with a .yaml file to run integration and dockerization of the app.py file automatically.


# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/FBrownp/ML-geo
```
### STEP 01- Create a new python venv environment after opening the repository

```bash
python -m venv venv
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### STEP 03- Init DVC and reproduce
```bash
dvc init
```
```bash
dvc repro
```

### STEP 04- run the app with the trained model

```bash 
python app.py
```


### DVC cmd

1. dvc init
2. dvc repro
3. dvc dag



## MLflow

[Documentation](https://mlflow.org/docs/latest/index.html)
[MLflow](https://dagshub.com/FBrownp/ML-geo.mlflow)

### dagshub
[dagshub](https://dagshub.com/)

