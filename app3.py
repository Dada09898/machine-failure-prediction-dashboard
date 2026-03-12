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
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- DARK THEME STYLE ---------------- #

st.markdown("""
<style>
.main {
background-color: #0e1117;
color: white;
}
.block-container {
padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

st.title("⚙️ Industrial Machine Failure Monitoring System")

st.write(
"""
AI powered monitoring system that predicts machine failure risk
and explains the decision using Explainable AI (SHAP).
"""
)

# ---------------- LOAD MODEL ---------------- #

model = joblib.load("models/best_model.pkl")
#explainer = shap.TreeExplainer(model)
explainer = shap.Explainer(model)

# ---------------- SIDEBAR INPUT PANEL ---------------- #

st.sidebar.title("Machine Control Panel")

machine_type = st.sidebar.selectbox(
    "Machine Type",
    ["Low Quality Machine (L)", "Medium Quality Machine (M)", "High Quality Machine (H)"]
)

air_temp = st.sidebar.slider("Air Temperature (K)", 295, 305, 300)
process_temp = st.sidebar.slider("Process Temperature (K)", 305, 315, 310)
rotational_speed = st.sidebar.slider("Rotational Speed (RPM)", 1100, 1800, 1500)
torque = st.sidebar.slider("Torque (Nm)", 20, 80, 40)
tool_wear = st.sidebar.slider("Tool Wear Time (min)", 0, 250, 10)

st.sidebar.subheader("Failure Indicators")

twf = st.sidebar.selectbox("Tool Wear Failure", [0,1])
hdf = st.sidebar.selectbox("Heat Dissipation Failure", [0,1])
pwf = st.sidebar.selectbox("Power Failure", [0,1])
osf = st.sidebar.selectbox("Overstrain Failure", [0,1])
rnf = st.sidebar.selectbox("Random Failure", [0,1])

predict = st.sidebar.button("Predict Machine Risk")

# ---------------- TYPE ENCODING ---------------- #

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

# ---------------- DASHBOARD TABS ---------------- #

    tab1, tab2, tab3 = st.tabs([
        "📊 Prediction Dashboard",
        "📈 Sensor Monitoring",
        "🧠 Explainable AI"
    ])

# =====================================================
# TAB 1 : PREDICTION DASHBOARD
# =====================================================

    with tab1:

        st.header("Machine Status")

        if prediction[0] == 1:
            st.error(f"⚠ High Risk of Failure ({probability*100:.2f}%)")
        else:
            st.success(f"✅ Machine Operating Normally ({(1-probability)*100:.2f}%)")

        risk = probability * 100

        if risk < 30:
            st.success(f"🟢 Health Status: GOOD")
        elif risk < 70:
            st.warning(f"🟡 Health Status: WARNING")
        else:
            st.error(f"🔴 Health Status: CRITICAL")

        st.subheader("Failure Risk Meter")

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk,
            title={'text': "Failure Risk %"},
            gauge={
                'axis': {'range': [0,100]},
                'steps':[
                    {'range':[0,30],'color':'green'},
                    {'range':[30,70],'color':'yellow'},
                    {'range':[70,100],'color':'red'}
                ]
            }
        ))

        col1,col2,col3 = st.columns([1,2,1])

        with col2:
            st.plotly_chart(gauge)

# =====================================================
# TAB 2 : SENSOR MONITORING
# =====================================================

    with tab2:

        st.header("Live Sensor Monitoring")

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

        st.subheader("Sensor Values")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=sensor_df["Sensor"],
            y=sensor_df["Value"],
            mode="lines+markers"
        ))

        st.plotly_chart(fig, width="stretch")

        st.subheader("Sensor Comparison")

        fig_bar = go.Figure()

        fig_bar.add_trace(go.Bar(
            x=sensor_df["Sensor"],
            y=sensor_df["Value"]
        ))

        st.plotly_chart(fig_bar)





# =====================================================
# TAB 3 : EXPLAINABLE AI
# =====================================================

    with tab3:

        st.header("Explainable AI (SHAP)")

        # ---------------- LOCAL SHAP VALUES ---------------- #

        shap_values = explainer.shap_values(data)

        if isinstance(shap_values, list):
            shap_values = shap_values[1][0]
        else:
            shap_values = shap_values[0,:,1]

        base_value = explainer.expected_value

        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1]

        shap_df = pd.DataFrame({
            "Feature":[
                "Machine Type",
                "Air Temp",
                "Process Temp",
                "RPM",
                "Torque",
                "Tool Wear",
                "TWF",
                "HDF",
                "PWF",
                "OSF",
                "RNF"
            ],
            "Impact":shap_values
        }).sort_values("Impact")

        col1, col2 = st.columns(2)

        # ---------------- FEATURE IMPORTANCE ---------------- #

        with col1:

            st.subheader("Feature Importance")

            fig_imp = go.Figure()

            fig_imp.add_trace(go.Bar(
                x=shap_df["Impact"],
                y=shap_df["Feature"],
                orientation="h"
            ))

            fig_imp.update_layout(
                title="Feature Impact on Prediction",
                xaxis_title="SHAP Value",
                yaxis_title="Feature"
            )

            st.plotly_chart(fig_imp, width="stretch")

            st.subheader("Top Failure Cause")

            top_feature = shap_df.iloc[-1]["Feature"]

            st.warning(
                f"⚠ Most influential factor affecting prediction: **{top_feature}**"
            )



            st.subheader("AI Maintenance Recommendation")

            recommendations = {
                "Tool Wear": "🔧 Replace cutting tool soon to avoid failure.",
                "Torque": "⚙ Check mechanical load and lubrication.",
                "Air Temp": "🌡 Improve cooling or ventilation.",
                "Process Temp": "🔥 Inspect thermal regulation system.",
                "RPM": "⚡ Check motor calibration and speed control.",
                "HDF": "❄ Cooling system may be failing.",
                "TWF": "🔩 Tool wear detected. Maintenance required.",
                "OSF": "⚠ Machine overstrain detected.",
                "PWF": "🔌 Power supply instability.",
                "RNF": "🔍 Random failure signal detected."
            }

            for key in recommendations:
                if key in top_feature:
                    st.info(recommendations[key])

        # ---------------- WATERFALL PLOT ---------------- #

        with col2:

            st.subheader("Local Explanation")

            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values,
                    base_values=base_value,
                    data=data.iloc[0],
                    feature_names=shap_df["Feature"]
                ),
                show=False
            )

            st.pyplot(plt.gcf())
            plt.clf()

        # ---------------- FORCE PLOT ---------------- #

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

        # ---------------- GLOBAL SHAP SUMMARY ---------------- #

         # st.subheader("SHAP Summary Plot (All Features)")

        # #df = pd.read_csv("data/train.csv")
        # sample_data = joblib.load("models/shap_sample.pkl")

        # #type_map = {"L":0, "M":1, "H":2}
        # #df["Type"] = df["Type"].map(type_map)

        # # X = df[[

        # #     "Type",
        # #     "Air temperature [K]",
        # #     "Process temperature [K]",
        # #     "Rotational speed [rpm]",
        # #     "Torque [Nm]",
        # #     "Tool wear [min]",
        # #     "TWF",
        # #     "HDF",
        # #     "PWF",
        # #     "OSF",
        # #     "RNF"

        # # ]]

        # #
        # # sample_data = X.sample(200)

        # # shap_values_full = explainer(sample_data)
        # # shap_values_full = shap_values_full.values

        # # if isinstance(shap_values_full, list):
        # #     shap_values_full = shap_values_full[1]

        # shap_values_full = explainer.shap_values(sample_data)

        # if isinstance(shap_values_full, list):
        #     shap_values_full = shap_values_full[1]
        # shap_values_full = np.array(shap_values_full)

        # fig = plt.figure()
        # st.write(sample_data.shape)
        # st.write(sample_data.columns)
        # shap.summary_plot(
        # shap_values_full,
        # sample_data,
        # plot_type="dot",
        # show=False
        # ) 

        # fig = plt.gcf()

        # st.pyplot(fig, width="stretch")
        # plt.clf()


        sample_data = joblib.load("models/shap_sample.pkl")

        # compute shap values
        shap_values_full = explainer(sample_data)

        # extract actual shap values array
        values = shap_values_full.values

        # binary classifier fix
        if values.ndim == 3:
            values = values[:, :, 1]

        plt.figure(figsize=(10,6))

        shap.summary_plot(
            values,
            sample_data,
            plot_type="dot",
            max_display=11,
            show=False
        )

        st.pyplot(plt.gcf())

        plt.clf()


        st.subheader("Feature Dependence Analysis")

        feature_name = st.selectbox(
            "Select Feature for Dependence Plot",
            list(sample_data.columns),
            #sample_data.columns
            key="dependence_feature"
        )

        #plt.figure(figsize=(8,5))
        fig, ax = plt.subplots(figsize=(8,5))

        shap.dependence_plot(
            feature_name,
            values,
            sample_data,
            interaction_index=None,
            ax=ax,
            show=False
        )

        st.pyplot(fig)
        plt.close(fig)
        st.subheader("Feature Interaction Heatmap")

        interaction_values = shap.TreeExplainer(model).shap_interaction_values(sample_data)

        # binary classifier case
        if isinstance(interaction_values, list):
            interaction_values = interaction_values[1]

        # take mean across samples
        interaction_matrix = np.abs(interaction_values).mean(axis=0)

        # ensure 2D matrix
        if interaction_matrix.ndim == 3:
            interaction_matrix = interaction_matrix[:,:,0]

        fig = plt.figure(figsize=(8,6))

        plt.imshow(interaction_matrix, cmap="coolwarm")
        plt.colorbar()

        plt.xticks(range(len(sample_data.columns)), sample_data.columns, rotation=90)
        plt.yticks(range(len(sample_data.columns)), sample_data.columns)

        plt.title("Feature Interaction Strength")

        st.pyplot(fig)

        plt.clf()