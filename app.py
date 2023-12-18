from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from mystic.solvers import diffev2
from mystic.math import almostEqual
from mystic.monitors import VerboseMonitor

def Model_prediction(data):

    model_1 = joblib.load("artifacts\model_trainer\model_1.joblib")
    model_2 = joblib.load("artifacts\model_trainer\model_2.joblib")


    def obj_func1(x, data = data ):
        data["Ps"] = x[0]
        return abs(model_1.predict(data)-1.5)

    def obj_func2(x, data = data ):
        data["Ps"] = x[0]
        return abs(model_2.predict(data)-1.5)   

    bds = [(10,50)]

    mon = VerboseMonitor(10)
    result1 = diffev2(obj_func1, x0=bds, npop=40,  gtol=30, disp=False, full_output=True, itermon=mon)
    mon = VerboseMonitor(10)
    result2 = diffev2(obj_func2, x0=bds, npop=40,  gtol=30, disp=False, full_output=True, itermon=mon)

    return result1[0],result2[0]


names = ["Cohesion", "Phi","Unit_weight","Pe","Ps","slope_angle","slope_height"]

class VariablesIn(BaseModel):
    Cohesion : float
    Phi : float
    Unit_weight : float
    Pe : float
    slope_angle : float
    slope_height : float

app = FastAPI()
@app.get("/")
def home():
    return {"Hello world"}

@app.post("/predict")
def predict(data : VariablesIn):
    data = list(data)
    new_data = [0,1,2,3,4,5,6]
    new_data[0] = data[0][1]
    new_data[1] = data[1][1]
    new_data[2] = data[2][1]
    new_data[3] = data[3][1]
    new_data[4] = 0
    new_data[5] = data[4][1]
    new_data[6] = data[5][1]
    input_data = pd.DataFrame([new_data], columns= names)
    

    p1,p2 = Model_prediction(input_data)

    p1 = np.array(p1).tolist()
    p2 = np.array(p2).tolist()


    return {"Pss" : p1 , "Psgmp" : p2}

if __name__ == "__main__":
    uvicorn.run(app , host="0.0.0.0", port=8080)