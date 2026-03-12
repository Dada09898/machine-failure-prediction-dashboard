import mlflow
import mlflow.sklearn
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from data_preprocessing import load_data


X_train,X_test,y_train,y_test = load_data("data/train.csv")

models = {

    "LogisticRegression":LogisticRegression(max_iter=1000),

    "DecisionTree":DecisionTreeClassifier(),

    "RandomForest":RandomForestClassifier(),

    "SVM":SVC()

}

best_accuracy = 0
best_model = None
best_model_name = ""

mlflow.set_experiment("Tata_Steel_Machine_Failure")

for name,model in models.items():

    with mlflow.start_run(run_name=name):

        model.fit(X_train,y_train)

        preds = model.predict(X_test)

        acc = accuracy_score(y_test,preds)

        mlflow.log_param("model",name)
        mlflow.log_metric("accuracy",acc)

        mlflow.sklearn.log_model(model,name)

        print(name,"Accuracy:",acc)

        if acc > best_accuracy:

            best_accuracy = acc
            best_model = model
            best_model_name = name


joblib.dump(best_model,"models/best_model.pkl")

print("Best Model:",best_model_name)
print("Best Accuracy:",best_accuracy)