import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="Industrial Machine Failure AI",
    page_icon="⚙️",
    layout="wide"
)

# ------------------------------------------------
# CSS
# ------------------------------------------------

st.markdown("""
<style>
body {
background-color:#0e1117;
}
.block-container {
padding-top:2rem;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# CACHED LOADING
# ------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("models/best_model.pkl")
    explainer = shap.Explainer(model)
    return model, explainer

model, explainer = load_model()

@st.cache_data
def load_sample():
    return joblib.load("models/shap_sample.pkl")

sample_data = load_sample()

# ------------------------------------------------
# TITLE
# ------------------------------------------------

st.title("⚙️ Industrial Machine Failure Monitoring System")

st.markdown("""
AI powered predictive maintenance system that monitors
machine sensors and predicts failure risk using ML.
""")

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

st.sidebar.title("Machine Control Panel")

# -------------------------------
# CSV DATA UPLOAD
# -------------------------------

# st.sidebar.subheader("Upload Machine Sensor Data")

# uploaded_file = st.sidebar.file_uploader(
# "Upload CSV file for batch prediction",
# type=["csv"]
# )


machine_type = st.sidebar.selectbox(
"Machine Type",
["Low Quality Machine (L)",
"Medium Quality Machine (M)",
"High Quality Machine (H)"]
)

air_temp = st.sidebar.slider("Air Temperature (K)",295,305,300)
process_temp = st.sidebar.slider("Process Temperature (K)",305,315,310)
rpm = st.sidebar.slider("Rotational Speed (RPM)",1100,1800,1500)
torque = st.sidebar.slider("Torque (Nm)",20,80,40)
tool_wear = st.sidebar.slider("Tool Wear Time (min)",0,250,10)

st.sidebar.subheader("Failure Indicators")

twf = st.sidebar.selectbox("Tool Wear Failure",[0,1])
hdf = st.sidebar.selectbox("Heat Dissipation Failure",[0,1])
pwf = st.sidebar.selectbox("Power Failure",[0,1])
osf = st.sidebar.selectbox("Overstrain Failure",[0,1])
rnf = st.sidebar.selectbox("Random Failure",[0,1])

# ------------------------------------------------
# PREDICT BUTTON
# ------------------------------------------------

if st.sidebar.button("Predict Machine Risk"):
    st.session_state["predict"] = True

# ------------------------------------------------
# RUN DASHBOARD
# ------------------------------------------------
# -------------------------------
# BATCH PREDICTION FROM CSV
# -------------------------------

# if uploaded_file is not None:

#     df = pd.read_csv(uploaded_file)

#     st.subheader("Uploaded Dataset")
#     st.dataframe(df)

#     st.download_button(
#     "Download Prediction Results",
#     df.to_csv(index=False),
#     file_name="machine_predictions.csv"
#     )

#     try:
#         df = df[features]

#         predictions = model.predict(df)
#         probabilities = model.predict_proba(df)[:,1]

#         df["Failure_Risk_%"] = probabilities * 100
#         df["Prediction"] = predictions

#         st.subheader("Prediction Results")

#         st.dataframe(df)

#         # Risk ranking
#         st.subheader("Highest Risk Machines")

#         top_risk = df.sort_values("Failure_Risk_%", ascending=False).head(10)

#         st.dataframe(top_risk)

#     except Exception as e:
#         st.error("CSV format incorrect. Please match training feature format.")
if st.session_state.get("predict", False):

    type_map = {
    "Low Quality Machine (L)":0,
    "Medium Quality Machine (M)":1,
    "High Quality Machine (H)":2
    }

    data = pd.DataFrame([[
    type_map[machine_type],
    air_temp,
    process_temp,
    rpm,
    torque,
    tool_wear,
    twf,
    hdf,
    pwf,
    osf,
    rnf
    ]],columns=[
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "TWF","HDF","PWF","OSF","RNF"
    ])

    features = joblib.load("models/model_features.pkl")
    data = data[features]

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    risk = probability * 100


    # -----------------------------
    # FAILURE ALERT
    # -----------------------------

    if risk > 70:
        st.error("🚨 Critical Failure Risk! Immediate maintenance required. High failure probability detected.")
    elif risk > 40:
        st.warning("⚠ Moderate Risk. Schedule maintenance soon. Risk level increasing.")
    else:
        st.success("✅Machine operating within safe limits.")

# ------------------------------------------------
# TABS
# ------------------------------------------------

    tab1,tab2,tab3,tab4 = st.tabs([
    "📊 Overview",
    "📈 Sensors",
    "🧠 Explainable AI",
    "🔬 Advanced Analysis"
    ])

# ------------------------------------------------
# TAB 1 OVERVIEW
# ------------------------------------------------

    with tab1:

        col1,col2,col3 = st.columns(3)

        col1.metric("Failure Risk %",f"{risk:.2f}%")

        if prediction == 1:
            col2.error("Machine Failure Risk Detected")
        else:
            col2.success("Machine Operating Normally")

        if risk < 30:
            col3.success("Health: GOOD")
        elif risk < 70:
            col3.warning("Health: WARNING")
        else:
            col3.error("Health: CRITICAL")

        gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk,
        title={'text':"Predicted Machine Failure Probability"},
        gauge={'axis':{'range':[0,100]}}
        ))

        st.plotly_chart(gauge,use_container_width=True)




        st.subheader("Simulated Machine Failure Risk Trend")

        timeline_data = pd.DataFrame({
        "Time":range(10),
        "Failure Risk":[risk + np.random.uniform(-5,5) for _ in range(10)]
        })

        fig_timeline = px.line(
        timeline_data,
        x="Time",
        y="Failure Risk",
        markers=True
        )

        st.plotly_chart(fig_timeline,use_container_width=True) 



       # -------------------------------
        # KPI CARDS
        # -------------------------------

        col1, col2, col3 = st.columns(3)

        col1.metric(
        label="Predicted Failure Risk",
        value=f"{risk:.2f}%"
        )

        col2.metric(
        label="Machine Status",
        value="Failure Risk" if prediction == 1 else "Normal"
        )

        col3.metric(
        label="Model Confidence",
        value=f"{probability*100:.2f}%"
        )
                

# ------------------------------------------------
# TAB 2 SENSOR MONITORING
# ------------------------------------------------

    with tab2:
         
        # -----------------------------
        # LIVE SENSOR SIMULATION
        # -----------------------------

        import random
        import time

        placeholder = st.empty()

        for _ in range(1):

            air_temp_live = air_temp + random.uniform(-1,1)
            rpm_live = rpm + random.uniform(-50,50)
            torque_live = torque + random.uniform(-2,2)

            live_df = pd.DataFrame({
                "Sensor":["Air Temp","RPM","Torque"],
                "Value":[air_temp_live,rpm_live,torque_live]
            })

            fig_live = px.bar(live_df,x="Sensor",y="Value",title="Simulated Live Sensor Monitoring ")

            placeholder.plotly_chart(fig_live,use_container_width=True)


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
        rpm,
        torque,
        tool_wear
        ]
        })

        fig1 = px.line(sensor_df,x="Sensor",y="Value",markers=True)
        st.plotly_chart(fig1,use_container_width=True)

        fig2 = px.bar(sensor_df,x="Sensor",y="Value",title="Current Machine Sensor Values")
        st.plotly_chart(fig2,use_container_width=True)

# ------------------------------------------------
# TAB 3 EXPLAINABLE AI
# ------------------------------------------------

    with tab3:

        shap_values = explainer(data)

        values = shap_values.values

        if values.ndim == 3:
            values = values[:,:,1]

        shap_row = values[0]

        base_value = shap_values.base_values
        if isinstance(base_value,np.ndarray):
            base_value = base_value.flatten()[0]

        shap_df = pd.DataFrame({
        "Feature":features,
        "Impact":shap_row
        }).sort_values("Impact")

        st.subheader("Explainable AI Root Cause Explanation")
        # -------------------------------
        # FAILURE REASON DETECTION
        # -------------------------------

        st.subheader("Primary Cause of Failure Risk")

        top_feature = shap_df.sort_values("Impact", ascending=False).iloc[0]["Feature"]

        st.warning(f"Main factor increasing machine failure risk: **{top_feature}**")

        top_feature = shap_df.iloc[-1]["Feature"]

        st.info(f"Primary factor affecting machine failure risk: {top_feature}")

        fig = px.bar(shap_df,x="Impact",y="Feature",orientation="h",title="Feature Contribution to Machine Failure Prediction")
        st.plotly_chart(fig,use_container_width=True)


        st.caption("**This chart shows how each sensor contributed to the model prediction.**")
        shap.plots.waterfall(
        shap.Explanation(
        values=shap_row,
        base_values=base_value,
        data=data.iloc[0],
        feature_names=features
        ),
        show=False
        )

        st.pyplot(plt.gcf())
        plt.clf()

       
        

        st.subheader("Prediction Explanation (SHAP Force Plot)")

        st.info(
        """
        This chart explains how each feature influenced the final prediction.

        🔴Red features increase machine failure risk.

        🔵Blue features decrease machine failure risk.

        The base value is the average model prediction and arrows show how each feature pushes the prediction higher or lower.
        """
        )
        
        force_plot = shap.force_plot(
        float(base_value),
        shap_row,
        data.iloc[0]
        )
        #st.success(f"Final model prediction score: {base_value + shap_row.sum():.2f}")
        components.html(
        shap.getjs()+force_plot.html(),
        height=400
        )
        st.success(f"Final model prediction score: {base_value + shap_row.sum():.2f}")
# ------------------------------------------------
# TAB 4 ADVANCED ANALYSIS
# ------------------------------------------------

    with tab4:

        shap_values_full = explainer(sample_data)

        values = shap_values_full.values

        if values.ndim == 3:
            values = values[:,:,1]

        st.subheader("Global SHAP Summary")

        plt.figure(figsize=(10,6))

        shap.summary_plot(values,sample_data,show=False)

        st.pyplot(plt.gcf())
        plt.clf()

# -------------------------------
# DEPENDENCE PLOT
# -------------------------------

        continuous_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
        ]
        st.markdown(
            "**This chart shows how a sensor value affects the machine failure prediction.**"
            )
        st.subheader("Feature Dependence")

        feature_name = st.selectbox(
        "Select Feature",
        continuous_features,
        key="dependence_feature"
        )

        rows = min(len(sample_data),len(values))

        sample_data_plot = sample_data.iloc[:rows]
        values_plot = values[:rows]

        fig, ax = plt.subplots(figsize=(7,4))

        shap.dependence_plot(
        feature_name,
        values_plot,
        sample_data_plot,
        interaction_index="auto",
        ax=ax,
        show=False
        )

        st.pyplot(fig)
        plt.close(fig)

# -------------------------------
# INTERACTION HEATMAP
# -------------------------------
        st.markdown(
            "**Red indicates strong interaction between features, Blue indicates weak interaction.**"
            )
        st.subheader("Feature Interaction Heatmap")

        interaction_values = shap.TreeExplainer(model).shap_interaction_values(sample_data)

        # binary classifier case
        if isinstance(interaction_values,list):
            interaction_values = interaction_values[1]

        interaction_matrix = np.abs(interaction_values).mean(axis=0)

       

        # 🔧 FIX
        if interaction_matrix.ndim == 3:
            interaction_matrix = interaction_matrix[:,:,0]

        fig, ax = plt.subplots(figsize=(8,6))

        plt.imshow(interaction_matrix,cmap="coolwarm")
        plt.colorbar()

        plt.xticks(range(len(sample_data.columns)),sample_data.columns,rotation=90)
        plt.yticks(range(len(sample_data.columns)),sample_data.columns)

        st.pyplot(fig)
        plt.close(fig)