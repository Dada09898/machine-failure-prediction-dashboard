import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit.components.v1 as components

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Industrial Machine Failure AI",
    layout="wide"
)

# ---------------- TITLE ---------------- #

st.title("⚙️ Industrial Machine Failure Monitoring System")

st.markdown(
"""
AI powered monitoring system that predicts machine failure risk
and explains the decision using Explainable AI (SHAP).
"""
)

# ---------------- LOAD MODEL ---------------- #

model = joblib.load("models/best_model.pkl")
explainer = shap.Explainer(model)

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("Machine Control Panel")

machine_type = st.sidebar.selectbox(
    "Machine Type",
    ["Low Quality Machine (L)", "Medium Quality Machine (M)", "High Quality Machine (H)"]
)

air_temp = st.sidebar.slider("Air Temperature (K)",295,305,300)
process_temp = st.sidebar.slider("Process Temperature (K)",305,315,310)
rotational_speed = st.sidebar.slider("Rotational Speed (RPM)",1100,1800,1500)
torque = st.sidebar.slider("Torque (Nm)",20,80,40)
tool_wear = st.sidebar.slider("Tool Wear Time (min)",0,250,10)

st.sidebar.subheader("Failure Indicators")

twf = st.sidebar.selectbox("Tool Wear Failure",[0,1])
hdf = st.sidebar.selectbox("Heat Dissipation Failure",[0,1])
pwf = st.sidebar.selectbox("Power Failure",[0,1])
osf = st.sidebar.selectbox("Overstrain Failure",[0,1])
rnf = st.sidebar.selectbox("Random Failure",[0,1])

predict = st.sidebar.button("Predict Machine Risk")

type_map = {
"Low Quality Machine (L)":0,
"Medium Quality Machine (M)":1,
"High Quality Machine (H)":2
}

# ---------------- PREDICTION ---------------- #

if predict:

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

    features = joblib.load("models/model_features.pkl")
    data = data[features]

    prediction = model.predict(data)
    probability = model.predict_proba(data)[0][1]

# ---------------- MAIN TABS ---------------- #

    tab1,tab2,tab3,tab4 = st.tabs([
        "📊 Overview",
        "📈 Sensors",
        "🧠 Explainable AI",
        "🔬 Advanced Analysis"
    ])

# ================= OVERVIEW ================= #

    with tab1:

        st.header("Prediction Overview")

        risk = probability*100

        col1,col2,col3 = st.columns(3)

        with col1:
            st.metric("Failure Risk %",f"{risk:.2f}%")

        with col2:
            if prediction[0]==1:
                st.error("Machine Failure Risk Detected")
            else:
                st.success("Machine Operating Normally")

        with col3:
            if risk<30:
                st.success("Health: GOOD")
            elif risk<70:
                st.warning("Health: WARNING")
            else:
                st.error("Health: CRITICAL")

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk,
            title={'text':"Failure Risk"},
            gauge={'axis':{'range':[0,100]}}
        ))

        st.plotly_chart(gauge,use_container_width=True)

# ================= SENSOR ================= #

    with tab2:

        st.header("Sensor Monitoring")

        sensor_df = pd.DataFrame({

            "Sensor":[
                "Air Temp",
                "Process Temp",
                "RPM",
                "Torque",
                "Tool Wear"
            ],

            "Value":[
                air_temp,
                process_temp,
                rotational_speed,
                torque,
                tool_wear
            ]
        })

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=sensor_df["Sensor"],
            y=sensor_df["Value"],
            mode="lines+markers"
        ))

        st.plotly_chart(fig,use_container_width=True)

        fig2 = go.Figure()

        fig2.add_trace(go.Bar(
            x=sensor_df["Sensor"],
            y=sensor_df["Value"]
        ))

        st.plotly_chart(fig2,use_container_width=True)

# ================= SHAP ================= #

    with tab3:

        st.header("Explainable AI")

        shap_values = explainer(data)

        values = shap_values.values

        # binary classifier fix
        if values.ndim == 3:
            values = values[:,:,1]

        # take first row (single prediction)
        shap_row = values[0]

        base_value = shap_values.base_values[0]

        col1,col2 = st.columns(2)

        with col1:

            st.subheader("Feature Importance")

            shap_df = pd.DataFrame({
                "Feature":features,
                "Impact":shap_row

            }).sort_values("Impact")

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=shap_df["Impact"],
                y=shap_df["Feature"],
                orientation="h"
            ))

            st.plotly_chart(fig,use_container_width=True)

        with col2:

            st.subheader("Waterfall Explanation")

            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values,
                    base_values=base_value,
                    data=data.iloc[0],
                    feature_names=features
                ),
                show=False
            )

            st.pyplot(plt.gcf())
            plt.clf()

        st.subheader("Force Plot")

        force_plot = shap.force_plot(
            base_value,
            shap_values,
            data.iloc[0]
        )

        components.html(
            shap.getjs()+force_plot.html(),
            height=300
        )

# ================= ADVANCED ================= #

    with tab4:

        st.header("Advanced SHAP Analysis")

        sample_data = joblib.load("models/shap_sample.pkl")

        shap_values_full = explainer(sample_data)

        values = shap_values_full.values

        if values.ndim==3:
            values = values[:,:,1]

        st.subheader("Global SHAP Summary")

        plt.figure(figsize=(10,6))

        shap.summary_plot(
            values,
            sample_data,
            show=False
        )

        st.pyplot(plt.gcf())
        plt.clf()

        st.subheader("Feature Dependence")

        feature_name = st.selectbox(
            "Select Feature",
            sample_data.columns
        )

        fig,ax = plt.subplots()

        shap.dependence_plot(
            feature_name,
            values,
            sample_data,
            ax=ax,
            show=False
        )

        st.pyplot(fig)

        st.subheader("Interaction Heatmap")

        interaction_values = shap.TreeExplainer(model).shap_interaction_values(sample_data)

        if isinstance(interaction_values,list):
            interaction_values = interaction_values[1]

        interaction_matrix = np.abs(interaction_values).mean(axis=0)

        if interaction_matrix.ndim==3:
            interaction_matrix = interaction_matrix[:,:,0]

        fig = plt.figure()

        plt.imshow(interaction_matrix,cmap="coolwarm")
        plt.colorbar()

        plt.xticks(range(len(sample_data.columns)),sample_data.columns,rotation=90)
        plt.yticks(range(len(sample_data.columns)),sample_data.columns)

        st.pyplot(fig)