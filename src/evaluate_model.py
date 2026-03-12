import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import roc_auc_score
from data_preprocessing import load_data

X_train,X_test,y_train,y_test = load_data("data/train.csv")

model = joblib.load("models/best_model.pkl")

pred_prob = model.predict_proba(X_test)[:,1]

roc = roc_auc_score(y_test,pred_prob)

print("ROC-AUC Score:",roc)

pred = model.predict(X_test)

print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, pred_prob)

plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()