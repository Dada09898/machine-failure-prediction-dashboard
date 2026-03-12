import joblib
import numpy as np

model = joblib.load("models/best_model.pkl")

data = np.array([[0,300,310,1500,40,5,0,0,0,0,0]])

pred = model.predict(data)

print(pred)