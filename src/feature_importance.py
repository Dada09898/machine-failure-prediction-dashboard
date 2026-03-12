import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load("models/best_model.pkl")

df = pd.read_csv("data/train.csv")

df = df.drop(["id","Product ID"],axis=1)

X = df.drop("Machine failure",axis=1)

importances = model.feature_importances_

features = X.columns

plt.figure(figsize=(8,5))

plt.barh(features,importances)

plt.xlabel("Importance")
plt.title("Feature Importance")

plt.show()