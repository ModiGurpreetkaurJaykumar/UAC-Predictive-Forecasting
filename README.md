# 📊 Predictive Forecasting of Care Load & Placement Demand

## 📌 Project Overview

This project applies time-series analysis and forecasting techniques to the **Unaccompanied Alien Children (UAC) Program dataset** to explore future care-load demand, discharge demand, and potential pressure on care capacity.

The goal is to move beyond historical reporting and provide forward-looking analysis that can help stakeholders explore possible changes in care demand.

An interactive **Streamlit dashboard** was developed to present historical trends, forecast outputs, model-comparison metrics, and illustrative scenarios.

> **Important:** Forecasts and scenario outputs are analytical estimates for this project. They are not official operational guidance or validated placement decisions.

---

## 🎯 Project Objectives

The project aims to answer the following questions:

- How many children may be under HHS care in the coming days or weeks?
- How might transfers into HHS compare with discharges from HHS care?
- What short-term discharge demand might be expected?
- When could forecast care load approach a user-specified capacity threshold?
- How do statistical and machine-learning forecasting approaches compare?
- How uncertain are the forecast results, and how does forecast error change with the horizon?

---

## 📊 Key Analyses

### 1. Historical Care Load Analysis

Examines historical changes in the number of children in HHS care and provides context for the forecast period.

### 2. Care Load Forecasting

Displays a forecast for future HHS care load using the selected forecasting method and forecast horizon.

### 3. Discharge Demand Forecasting

Explores expected short-term discharge volumes using the available discharge forecasting methods.

### 4. Model Comparison and Evaluation

Compares the available forecasting models using evaluation metrics such as:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

The evaluation results should be interpreted in the context of the validation method and forecast horizon used.

### 5. Intake and Exit Pressure Analysis

Compares transfers out of CBP custody into HHS care with children discharged from HHS care. The net-pressure indicator is calculated as:

**Net Pressure = Transfers − Discharges**

This is a flow indicator and does not fully account for every factor that can change the active HHS care load.

### 6. Illustrative Scenario Comparison

Allows users to explore an illustrative transfer-increase assumption and compare the scenario with the base forecast. This is a sensitivity illustration, not a causal estimate or independently validated forecast.

### 7. Capacity Stress Assessment

Allows a user to enter a capacity threshold and examine forecast values in relation to that threshold. Use an approved capacity value when making any real-world interpretation.

### 8. Forecast Uncertainty

Displays an approximate uncertainty band where available. Its interpretation depends on how the band was calculated and validated.

---

## 📈 Interactive Streamlit Dashboard

An interactive dashboard was developed using **Streamlit** to make the analysis accessible through charts, summary indicators, and sidebar controls.

The dashboard includes:

- Care-load forecast chart
- Discharge-demand forecast chart
- Forecast horizon selector
- Care-load model selector
- Discharge forecast method selector
- Historical time-window selector
- Approximate uncertainty-band toggle
- Capacity threshold input
- Illustrative transfer-increase control
- Model-performance comparison
- Intake and exit pressure analysis
- Scenario comparison visualization
- Forecast results table and CSV download

---

## 📊 Dashboard Modules

### 📈 Care Load Forecast

Displays historical HHS care load alongside the selected model's forecast for the chosen horizon.

### 🏠 Discharge Demand Forecast

Displays historical HHS discharges and the selected discharge forecast.

### 📉 Model Performance Comparison

Presents the available model-comparison results using metrics such as MAE, RMSE, and MAPE.

### 🔄 Intake and Exit Pressure

Compares daily transfers into HHS care with HHS discharges and visualizes the difference between these flows.

### ⚠️ Capacity Stress Assessment

Compares forecast care-load values with the capacity threshold entered by the user, when one is supplied.

### 🧪 Scenario Comparison

Compares the base forecast with an illustrative scenario based on the selected transfer-increase assumption.

### 📥 Forecast Results Export

Displays forecast output and provides a CSV download for further analysis.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Plotly
- Streamlit
- Scikit-learn
- Statsmodels
- Google Colab
- Jupyter Notebook
- Visual Studio Code
- Git
- GitHub

The exact dependencies required to run the application are listed in `requirements.txt`.

---

## 📁 Project Structure

```text
UAC_Predictive_Forecasting/
│
├── Data/
│   ├── HHS_Unaccompanied_Alien_Children_Program.csv
│   ├── uac_daily_cleaned.csv
│   ├── model_comparison_results.csv
│   └── final_care_load_forecast.csv
│
├── Notebooks/
│   └── Predictive_Forecasting_Care_Load_and_Placement_Demand.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

File names shown above reflect the files used during development. Confirm that the names match your local project before relying on this structure.

---

## 📂 Dataset

The project uses daily reporting data for the UAC Program. The principal fields described in the project instructions include:

- **Date:** Reporting date
- **Children apprehended and placed in CBP custody:** Daily intake volume
- **Children in CBP custody:** Active CBP care load
- **Children transferred out of CBP custody:** Flow into the HHS system
- **Children in HHS Care:** Active HHS care load
- **Children discharged from HHS Care:** Successful sponsor placements

The cleaned dataset and derived CSV files support the analysis and dashboard. Confirm dataset provenance and sharing permissions before publishing any data files.

---

## 🧹 Data Preparation and Feature Engineering

The project methodology includes the following preparation and feature-engineering tasks:

- Convert the reporting date to a datetime index
- Check continuity of daily observations
- Identify and handle missing dates or values
- Examine trend, seasonality, and residual patterns
- Create lag features using previous observations
- Calculate 7-day and 14-day rolling averages and variances
- Calculate the flow-based net-pressure indicator
- Consider calendar effects such as day of week and month
- Use time-ordered training and testing data
- Evaluate forecasts at multiple horizons

The analysis notebook in the `Notebooks/` directory documents the analytical workflow.

---

## 📐 Forecasting Models and Evaluation Metrics

### Forecasting Models

The project instructions specify comparison of the following approaches:

- **Naive Persistence:** Uses the most recent observed value as the forecast.
- **Moving Average:** Uses recent observations to estimate future values.
- **ARIMA:** Statistical time-series model for autoregressive and moving-average patterns.
- **SARIMA:** Statistical time-series model that can represent seasonal patterns.
- **Exponential Smoothing:** Forecasting method that weights historical observations according to the chosen specification.
- **Random Forest Regressor:** Machine-learning model based on an ensemble of decision trees.
- **Gradient Boosting Regressor:** Machine-learning model that builds an ensemble sequentially.

### Evaluation Metrics

**MAE — Mean Absolute Error**

Measures the average absolute difference between actual and forecast values. Lower values are generally better.

**RMSE — Root Mean Squared Error**

Penalizes larger forecast errors more strongly. Lower values are generally better.

**MAPE — Mean Absolute Percentage Error**

Expresses forecast errors as percentages. It requires care when actual values are zero or close to zero.

**Forecast Horizon Error**

Compares forecast performance across different horizons, including 1-day, 7-day, and 14-day forecasts where evaluated.

### Validation Approach

The project instructions call for:

- A strict time-based train-test split
- Walk-forward validation
- Multi-horizon evaluation
- Comparison of statistical and machine-learning approaches

The stored comparison results should be treated as preliminary until the validation procedure, horizon definitions, and metric calculations have been verified consistently for every model.

---

## 🚀 Running the Dashboard Locally

### 1. Prerequisites

Install Python and Visual Studio Code. A virtual environment is recommended.

### 2. Open the Project Folder

Open `UAC_Predictive_Forecasting` in VS Code and open a terminal in that directory.

### 3. Create and Activate a Virtual Environment (Optional)

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If the environment is already created and activated, you can skip this step.

### 4. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the Streamlit Application

```powershell
python -m streamlit run app.py
```

Open the local URL shown in the terminal. It is commonly:

```text
http://localhost:8501
```

To stop the dashboard, focus the terminal and press `Ctrl+C`.

---

## 🧭 How to Use the Dashboard

1. Choose a forecast horizon.
2. Select a care-load forecasting model.
3. Choose the historical time window.
4. Toggle the approximate uncertainty band if available.
5. Enter a capacity threshold only when you have a suitable value.
6. Adjust the illustrative transfer-increase percentage to explore a scenario.
7. Select a discharge forecast method.
8. Review the care-load and discharge forecasts, model metrics, flow analysis, and scenario comparison.
9. Download the forecast results CSV if you need a local copy.

Control behavior depends on the implementation in `app.py` and the data available to the application.

---

## 📦 Project Data Files

The `Data/` directory contains the original or cleaned dataset and derived outputs used by the notebook and dashboard.

- **`HHS_Unaccompanied_Alien_Children_Program.csv`** — Original dataset, if included under this exact name.
- **`uac_daily_cleaned.csv`** — Cleaned daily dataset used by the analysis or dashboard.
- **`model_comparison_results.csv`** — Exported model-comparison results.
- **`final_care_load_forecast.csv`** — Exported care-load forecast output.

Confirm the exact file names and how `app.py` uses each file. Do not publish data unless you have permission to share it.

---

## ⚠️ Limitations and Responsible Use

- Forecast performance depends on the quality, coverage, and characteristics of the available historical data.
- Sudden policy changes, border activity, humanitarian events, and other external factors may not be represented in the model.
- A flat forecast is expected from a Naive Persistence model when the latest observed value is carried forward unchanged.
- Approximate uncertainty bands should not be interpreted as guaranteed bounds unless their statistical construction and coverage have been validated.
- The illustrative transfer scenario is a sensitivity analysis, not a causal estimate or an independently validated prediction.
- A transfers-minus-discharges indicator does not fully explain all changes in active HHS care load.
- Capacity thresholds must not be presented as official limits unless supported by an authoritative source.
- Model results should not be used as the sole basis for operational decisions affecting children.
- Before uploading to a public GitHub repository, verify that the original dataset and all derived files are authorized for public release. Remove confidential, restricted, or personal information where applicable.

---

## 🔮 Future Improvements

- Strengthen walk-forward validation for every model
- Verify MAE, RMSE, and MAPE calculations across all forecast horizons
- Add model diagnostics and residual analysis
- Validate uncertainty intervals against held-out observations
- Improve missing-date and missing-value reporting
- Add automated tests for data loading, forecast generation, and CSV export
- Document data provenance and the data dictionary
- Add a deployment guide and reproducible environment setup

---

## 🌐 Dashboard Deployment

The interactive dashboard is built using **Streamlit** and can be published through Streamlit Community Cloud.

**Live Dashboard:** 

---

## 🔗 GitHub Repository

The GitHub repository will contain the project source code, analysis notebook, dataset files where appropriate, dependencies, and documentation.

**Repository URL:** 

---

## 📝 Conclusion

This project explores how daily UAC reporting data can be extended from historical reporting to predictive analysis. By combining care-load forecasting, discharge-demand forecasting, model comparison, flow-based indicators, uncertainty visualization, and illustrative scenarios, the dashboard provides a way to examine potential future demand.

The results should be interpreted with attention to model validation, uncertainty, data limitations, and responsible use. Further validation is required before any forecast is considered suitable for operational decision-making.

---

## 👤 Project Type

**Time-Series Forecasting and Data Analytics Project**

**Project:** Predictive Forecasting of Care Load & Placement Demand
