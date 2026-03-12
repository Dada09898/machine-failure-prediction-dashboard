import mlflow
import mlflow.sklearn
import joblib

model = joblib.load("models/best_model.pkl")

with mlflow.start_run():

    mlflow.sklearn.log_model(

        sk_model=model,
        name="tata_failure_model",
        registered_model_name="TataSteelFailureModel"

    )