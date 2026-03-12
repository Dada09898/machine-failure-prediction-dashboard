# Industrial Machine Failure Prediction Dashboard

This project is a machine learning based dashboard that predicts the probability of machine failure using industrial sensor data.

The system analyzes machine parameters such as temperature, rotational speed, torque and tool wear to estimate the risk of failure.

## Features

* Machine failure prediction
* Interactive dashboard using Streamlit
* Sensor monitoring visualization
* SHAP based explainable AI
* Feature importance analysis
* Feature interaction heatmap

## Tech Stack

* Python
* Streamlit
* Scikit-learn
* SHAP
* Plotly
* MLflow

## Project Structure

```
machine-failure-prediction-dashboard
│
├── app.py
├── models/
├── src/
├── data/
├── requirements.txt
└── README.md
```

## How to Run

Clone the repository:

```
git clone https://github.com/Dada09898/machine-failure-prediction-dashboard.git
```

Go to the project folder:

```
cd machine-failure-prediction-dashboard
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the dashboard:

```
streamlit run app.py
```

## Author

Ankit Kumar Singh
