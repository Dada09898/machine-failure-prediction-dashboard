import shap
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# load model
model = joblib.load("models/best_model.pkl")

# load dataset
df = pd.read_csv("data/train.csv")

# drop unnecessary columns
df = df.drop(["id","Product ID"], axis=1)

# encode Type column
le = LabelEncoder()
df["Type"] = le.fit_transform(df["Type"])

# features
X = df.drop("Machine failure", axis=1)
# save sample for shap dashboard
joblib.dump(X.sample(500),"models/shap_sample.pkl")

# create explainer
explainer = shap.TreeExplainer(model)
#explainer = shap.Explainer(model)
# shap values
shap_values = explainer.shap_values(X)

# plot
shap.summary_plot(shap_values, X)