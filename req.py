import requests

var = {
  "Cohesion": 0,
  "Phi": 35,
  "Unit_weight": 17,
  "Pe": 20,
  "slope_angle": 90,
  "slope_height": 15
}
r = requests.post('http://127.0.0.1:8000/predict', json=var).json()
