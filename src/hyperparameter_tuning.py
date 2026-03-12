import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from data_preprocessing import load_data

X_train,X_test,y_train,y_test = load_data("data/train.csv")

param_grid = {

    "n_estimators":[100,200],
    "max_depth":[5,10,20],
    "min_samples_split":[2,5]

}

model = RandomForestClassifier()

grid = GridSearchCV(

    model,
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1

)

grid.fit(X_train,y_train)

best_model = grid.best_estimator_

print("Best Parameters:",grid.best_params_)

joblib.dump(best_model,"models/best_model.pkl")
# save feature names
joblib.dump(X_train.columns,"models/model_features.pkl")
