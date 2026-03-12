import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
def load_data(path):

    df = pd.read_csv(path)

    df = df.drop(["id","Product ID"],axis=1)

    le = LabelEncoder()
    df["Type"] = le.fit_transform(df["Type"])
    
    joblib.dump(le,"models/type_encoder.pkl")

    X = df.drop("Machine failure",axis=1)
    y = df["Machine failure"]

    return train_test_split(
        X,y,test_size=0.2,random_state=42
    )