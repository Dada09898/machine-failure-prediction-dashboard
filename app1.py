import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd

# Load model
model = joblib.load("models/best_model.pkl")

# SHAP explainer
explainer = shap.TreeExplainer(model)

st.title("Tata Steel Machine Failure Prediction")

# ---------------- INPUTS ---------------- #

type_val = st.selectbox("Machine Type", ["L", "M", "H"])

air = st.number_input("Air temperature [K]")
process = st.number_input("Process temperature [K]")
speed = st.number_input("Rotational speed [rpm]")
torque = st.number_input("Torque [Nm]")
wear = st.number_input("Tool wear [min]")

TWF = st.number_input("TWF")
HDF = st.number_input("HDF")
PWF = st.number_input("PWF")
OSF = st.number_input("OSF")
RNF = st.number_input("RNF")

type_map = {"L":0,"M":1,"H":2}

feature_names = [
"Type",
"Air temperature [K]",
"Process temperature [K]",
"Rotational speed [rpm]",
"Torque [Nm]",
"Tool wear [min]",
"TWF",
"HDF",
"PWF",
"OSF",
"RNF"
]

# ---------------- PREDICT ---------------- #

if st.button("Predict"):

    data = pd.DataFrame([[
        type_map[type_val],
        air,
        process,
        speed,
        torque,
        wear,
        TWF,
        HDF,
        PWF,
        OSF,
        RNF
    ]], columns=feature_names)

    pred = model.predict(data)
    prob = model.predict_proba(data)[0][1]

    if pred[0] == 1:
        st.error(f"⚠ Machine Failure Risk ({prob*100:.2f}% probability)")
    else:
        st.success(f"✅ Machine Safe ({(1-prob)*100:.2f}% confidence)")

    # ---------------- SHAP ---------------- #

    # shap_values = explainer.shap_values(data)

    # if isinstance(shap_values, list):
    #     shap_values = shap_values[1]

    # shap_values = shap_values[0]

    # -------- Feature Importance -------- #
    # shap_values = explainer(data)
    # st.subheader("Feature Importance")

    # shap.plots.bar(
    #     shap.Explanation(
    #         values=shap_values,
    #         base_values=explainer.expected_value,
    #         data=data.iloc[0],
    #         feature_names=feature_names
    #     ),
    #     show=False
    # )

    # st.pyplot(plt.gcf())
    # plt.clf()

    # # -------- Waterfall Plot -------- #

    # st.subheader("Local Explanation")

    # shap.plots.waterfall(
    #     shap.Explanation(
    #         values=shap_values,
    #         base_values=explainer.expected_value,
    #         data=data.iloc[0],
    #         feature_names=feature_names
    #     ),
    #     show=False
    # )

    # st.pyplot(plt.gcf())
    # plt.clf()

    # # -------- Force Plot -------- #

    # st.subheader("Force Plot")

    # fig = shap.force_plot(
    #     explainer.expected_value,
    #     shap_values,
    #     data.iloc[0],
    #     matplotlib=True
    # )

    # st.pyplot(fig)

    shap_values = explainer.shap_values(data)

    # For RandomForest binary classifier
    if isinstance(shap_values, list):
        shap_values = shap_values[1][0]

    # For numpy array output
    else:
        shap_values = shap_values[0,:,1]
   # -------- Feature Importance -------- #

    st.subheader("Feature Importance")

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "value": shap_values
    }).sort_values("value")

    fig, ax = plt.subplots()

    ax.barh(shap_df["feature"], shap_df["value"])

    ax.set_xlabel("SHAP Value")
    ax.set_title("Feature Impact on Prediction")

    st.pyplot(fig)
    plt.close()

    # -------- Waterfall Plot -------- #

    st.subheader("Local Explanation")

    base_value = explainer.expected_value

    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1]

    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values,
            base_values=base_value,
            data=data.iloc[0],
            feature_names=feature_names
        ),
        show=False
    )

    st.pyplot(plt.gcf())
    plt.clf()

    # -------- Force Plot -------- #

    st.subheader("Force Plot")

    fig = shap.force_plot(
        base_value,
        shap_values,
        data.iloc[0],
        matplotlib=True
    )
    st.pyplot(fig)