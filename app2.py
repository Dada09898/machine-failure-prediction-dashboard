import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd
import streamlit.components.v1 as components
import plotly.graph_objects as go
# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Tata Steel Machine Failure Prediction",
    layout="wide"
)

st.title("Tata Steel Industrial Machine Failure Prediction System")

st.write(
"""
This AI system predicts the risk of industrial machine failure using
sensor data from manufacturing equipment.

It also provides Explainable AI (SHAP) insights to understand
which parameters influence the prediction.
"""
)

# ---------------- LOAD MODEL ---------------- #

model = joblib.load("models/best_model.pkl")

explainer = shap.TreeExplainer(model)

# ---------------- FEATURE NAMES ---------------- #

feature_names = [

"Machine Type",

"Air Temperature (Kelvin)",

"Process Temperature (Kelvin)",

"Rotational Speed (RPM)",

"Torque (Newton Meter)",

"Tool Wear Time (Minutes)",

"Tool Wear Failure",

"Heat Dissipation Failure",

"Power Failure",

"Overstrain Failure",

"Random Failure"

]

# ---------------- INPUT SECTION ---------------- #

st.header("Machine Sensor Inputs")

col1, col2 = st.columns(2)

with col1:

    machine_type = st.selectbox(
        "Machine Type",
        ["Low Quality Machine (L)", "Medium Quality Machine (M)", "High Quality Machine (H)"]
    )

    air_temp = st.slider(
        "Air Temperature (Kelvin)",
        295, 305, 300
    )

    process_temp = st.slider(
        "Process Temperature (Kelvin)",
        305, 315, 310
    )

    rotational_speed = st.slider(
        "Rotational Speed (RPM)",
        1100, 1800, 1500
    )

    torque = st.slider(
        "Torque (Newton Meter)",
        20, 80, 40
    )

with col2:

    tool_wear = st.slider(
        "Tool Wear Time (Minutes)",
        0, 250, 10
    )

    twf = st.selectbox(
        "Tool Wear Failure (TWF)",
        [0,1],
        help="Failure due to excessive tool wear"
    )

    hdf = st.selectbox(
        "Heat Dissipation Failure (HDF)",
        [0,1],
        help="Failure caused by insufficient heat dissipation"
    )

    pwf = st.selectbox(
        "Power Failure (PWF)",
        [0,1],
        help="Failure caused by power overload"
    )

    osf = st.selectbox(
        "Overstrain Failure (OSF)",
        [0,1],
        help="Failure due to mechanical overstrain"
    )

    rnf = st.selectbox(
        "Random Failure (RNF)",
        [0,1],
        help="Unexpected random machine failure"
    )

# ---------------- TYPE ENCODING ---------------- #

type_map = {
"Low Quality Machine (L)":0,
"Medium Quality Machine (M)":1,
"High Quality Machine (H)":2
}

# ---------------- PREDICTION ---------------- #

if st.button("Predict Machine Failure Risk"):

    data = pd.DataFrame([[

    type_map[machine_type],
    air_temp,
    process_temp,
    rotational_speed,
    torque,
    tool_wear,
    twf,
    hdf,
    pwf,
    osf,
    rnf

    ]], columns=[

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

    ])

    prediction = model.predict(data)

    probability = model.predict_proba(data)[0][1]

    st.header("Prediction Result")

    if prediction[0] == 1:

        st.error(
            f"⚠ High Risk of Machine Failure ({probability*100:.2f}% probability)"
        )

    else:

        st.success(
            f"✅ Machine Operating Normally ({(1-probability)*100:.2f}% confidence)"
        )


    st.subheader("Machine Health Status")

    risk = probability * 100

    if risk < 30:
        st.success(f"🟢 Machine Health: GOOD ({risk:.2f}% risk)")
    elif risk < 70:
        st.warning(f"🟡 Machine Health: WARNING ({risk:.2f}% risk)")
    else:
        st.error(f"🔴 Machine Health: CRITICAL ({risk:.2f}% risk)")





    st.subheader("Machine Failure Risk Meter")

    risk_percentage = probability * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_percentage,
        title={'text': "Failure Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
        }
    ))
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        
       st.plotly_chart(fig)   






    st.subheader("Sensor Monitoring Overview")

    sensor_data = pd.DataFrame({
        "Sensor": [
            "Air Temperature",
            "Process Temperature",
            "Rotational Speed",
            "Torque",
            "Tool Wear"
        ],
        "Value": [
            air_temp,
            process_temp,
            rotational_speed,
            torque,
            tool_wear
        ]
    })

    fig, ax = plt.subplots()

    ax.bar(sensor_data["Sensor"], sensor_data["Value"])

    ax.set_ylabel("Sensor Value")
    ax.set_title("Machine Sensor Readings")

    plt.xticks(rotation=45)

    st.pyplot(fig)

    plt.close(fig)
# ---------------- SHAP EXPLANATION ---------------- #

    st.header("Explainable AI Insights (SHAP)")

    shap_values = explainer.shap_values(data)

    if isinstance(shap_values, list):

        shap_values = shap_values[1][0]

    else:

        shap_values = shap_values[0,:,1]

    base_value = explainer.expected_value

    if isinstance(base_value,(list,np.ndarray)):

        base_value = base_value[1]

# ---------------- FEATURE IMPORTANCE ---------------- #

    st.subheader("Feature Importance")

    shap_df = pd.DataFrame({

        "Feature":feature_names,

        "Impact":shap_values

    }).sort_values("Impact")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.barh(shap_df["Feature"], shap_df["Impact"])

    ax.set_xlabel("SHAP Impact")

    ax.set_title("Feature Contribution to Prediction")

    st.pyplot(fig)
    


    st.subheader("Top Failure Cause")

    top_feature = shap_df.iloc[-1]["Feature"]
    top_value = shap_df.iloc[-1]["Impact"]

    st.info(
        f"The most influential factor affecting the prediction is **{top_feature}** "
        f"with SHAP impact value **{top_value:.3f}**."
    )
# ---------------- WATERFALL PLOT ---------------- #

    st.subheader("Local Prediction Explanation")

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

# ---------------- FORCE PLOT ---------------- #

    # -------- SHAP FORCE PLOT (Interactive) -------- #

    st.subheader("SHAP Force Plot")

    force_plot = shap.force_plot(
        base_value,
        shap_values,
        data.iloc[0]
    )

    components.html(
        shap.getjs() + force_plot.html(),
        height=350
    )