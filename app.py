
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="UAC Care Demand Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
DATA_FILE = DATA_DIR / "uac_daily_cleaned.csv"
RESULTS_FILE = DATA_DIR / "model_comparison_results.csv"
# ============================================================
# 2. DASHBOARD DESIGN
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #F4F7FB;
        color: #182B49;
    }

    [data-testid="stHeader"] {
        background: #F4F7FB;
    }

    [data-testid="stSidebar"] {
        background: #172B4D;
    }

    
[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #172B4D !important;
}

[data-testid="stSidebar"] input {
    color: #172B4D !important;
    -webkit-text-fill-color: #172B4D !important;
}


    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        background: #263D60;
    }

    .block-container {
    max-width: 1600px;
    padding-top: 3rem;
    padding-bottom: 2rem;
    }

.dashboard-title {
    font-size: 30px;
    font-weight: 750;
    line-height: 1.4;
    color: #172B4D;
    margin-top: 12px;
    margin-bottom: 8px;
    overflow-wrap: anywhere;
    }

    .dashboard-subtitle {
        font-size: 13px;
        color: #77849A;
        margin-bottom: 24px;
    }

    .section-title {
        font-size: 19px;
        font-weight: 700;
        color: #172B4D;
        margin: 10px 0 12px 0;
    }

    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E5EAF2;
        border-radius: 12px;
        padding: 17px 18px;
        box-shadow: 0 3px 12px rgba(25, 45, 75, 0.04);
        min-height: 110px;
    }

    [data-testid="stMetricLabel"] {
        color: #758198;
        font-size: 12px;
    }

    [data-testid="stMetricValue"] {
        color: #172B4D;
        font-size: 27px;
        font-weight: 700;
    }

    div[data-testid="stPlotlyChart"] {
        background: #FFFFFF;
        border: 1px solid #E5EAF2;
        border-radius: 12px;
        padding: 8px;
        box-shadow: 0 3px 12px rgba(25, 45, 75, 0.035);
    }

    div[data-testid="stDataFrame"] {
        background: #FFFFFF;
        border-radius: 10px;
    }

    .info-note {
        background: #EAF1FC;
        color: #29466E;
        padding: 12px 15px;
        border-radius: 9px;
        font-size: 12px;
        margin: 8px 0 18px 0;
    }

    hr {
        border-color: #E4EAF2;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 3. LOAD AND PREPARE DATA
# ============================================================

@st.cache_data
def load_dataset(path):
    data = pd.read_csv(path)

    if "Date" not in data.columns:
        data = data.rename(columns={data.columns[0]: "Date"})

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])
    data = data.sort_values("Date")

    # Remove time components and combine any duplicate dates.
    data["Date"] = data["Date"].dt.normalize()

    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

    for col in data.columns:
        if col != "Date" and col not in numeric_cols:
            data[col] = pd.to_numeric(
                data[col].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            )

    data = data.groupby("Date").mean(numeric_only=True)
    data = data.sort_index().asfreq("D")
    data = data.interpolate(method="time").ffill().bfill()

    return data


if not DATA_FILE.exists():
    st.error(
        "Dataset not found. Place uac_daily_cleaned.csv "
        "inside the Data folder."
    )
    st.stop()

df = load_dataset(DATA_FILE)

CARE_COL = "Children in HHS Care"
DISCHARGE_COL = "Children discharged from HHS Care"
TRANSFER_COL = "Children transferred out of CBP custody"

required = [CARE_COL, DISCHARGE_COL, TRANSFER_COL]
missing = [col for col in required if col not in df.columns]

if missing:
    st.error(f"Missing required dataset columns: {missing}")
    st.stop()

care = df[CARE_COL].astype(float)
discharges = df[DISCHARGE_COL].astype(float)
transfers = df[TRANSFER_COL].astype(float)

net_pressure = transfers - discharges

# ============================================================
# 4. FORECASTING FUNCTIONS
# ============================================================

def basic_forecast(series, horizon, model_name):
    series = series.astype(float).dropna()

    if model_name == "Naive Persistence":
        return np.repeat(series.iloc[-1], horizon).astype(float)

    if model_name == "Moving Average":
        return np.repeat(series.tail(7).mean(), horizon).astype(float)

    if model_name == "ARIMA":
        try:
            fit = ARIMA(series, order=(1, 1, 1)).fit()
            return np.asarray(fit.forecast(steps=horizon), dtype=float)
        except Exception:
            return np.repeat(series.iloc[-1], horizon).astype(float)

    if model_name == "Exponential Smoothing":
        try:
            fit = ExponentialSmoothing(
                series,
                trend="add",
                seasonal=None,
                initialization_method="estimated",
            ).fit()
            return np.asarray(fit.forecast(horizon), dtype=float)
        except Exception:
            return np.repeat(series.iloc[-1], horizon).astype(float)

    if model_name == "SARIMA":
        try:
            fit = SARIMAX(
                series,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            return np.asarray(fit.forecast(horizon), dtype=float)
        except Exception:
            return np.repeat(series.iloc[-1], horizon).astype(float)

    raise ValueError(f"Unsupported model: {model_name}")


def build_features(series):
    x = pd.DataFrame(index=series.index)

    x["lag_1"] = series.shift(1)
    x["lag_7"] = series.shift(7)
    x["lag_14"] = series.shift(14)
    x["mean_7"] = series.shift(1).rolling(7).mean()
    x["mean_14"] = series.shift(1).rolling(14).mean()
    x["variance_7"] = series.shift(1).rolling(7).var()
    x["variance_14"] = series.shift(1).rolling(14).var()
    x["day_of_week"] = x.index.dayofweek
    x["month"] = x.index.month

    return x


def ml_forecast(series, horizon, model_name):
    series = series.astype(float).copy()

    features = build_features(series)
    training = features.join(series.rename("target")).dropna()

    if len(training) < 30:
        return np.repeat(series.iloc[-1], horizon).astype(float)

    X = training.drop(columns="target")
    y = training["target"]

    if model_name == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        )

    model.fit(X, y)

    history = series.copy()
    predictions = []

    for _ in range(horizon):
        next_date = history.index[-1] + pd.Timedelta(days=1)

        row = pd.DataFrame(index=[next_date])
        row["lag_1"] = history.iloc[-1]
        row["lag_7"] = history.iloc[-7]
        row["lag_14"] = history.iloc[-14]
        row["mean_7"] = history.tail(7).mean()
        row["mean_14"] = history.tail(14).mean()
        row["variance_7"] = history.tail(7).var()
        row["variance_14"] = history.tail(14).var()
        row["day_of_week"] = next_date.dayofweek
        row["month"] = next_date.month
        row = row[X.columns]

        prediction = float(model.predict(row)[0])
        predictions.append(prediction)
        history.loc[next_date] = prediction

    return np.asarray(predictions)


def forecast(series, horizon, model_name):
    if model_name in ["Random Forest", "Gradient Boosting"]:
        return ml_forecast(series, horizon, model_name)

    return basic_forecast(series, horizon, model_name)


# ============================================================
# 5. SIDEBAR CONTROLS
# ============================================================

st.sidebar.markdown("## UAC Analytics")
st.sidebar.caption("Care Load & Placement Demand")
st.sidebar.divider()

st.sidebar.markdown("### Forecast Controls")

horizon = st.sidebar.slider(
    "Forecast horizon (days)",
    min_value=1,
    max_value=30,
    value=7,
)

model_name = st.sidebar.selectbox(
    "Care-load forecasting model",
    [
        "Naive Persistence",
        "Moving Average",
        "ARIMA",
        "Exponential Smoothing",
        "SARIMA",
        "Random Forest",
        "Gradient Boosting",
    ],
)

history_days = st.sidebar.selectbox(
    "Historical view",
    [30, 90, 180, 365, "All"],
    index=2,
)

show_interval = st.sidebar.checkbox(
    "Show approximate uncertainty band",
    value=True,
)

capacity = st.sidebar.number_input(
    "Care capacity threshold (optional)",
    min_value=0,
    value=0,
    step=100,
)

transfer_increase = st.sidebar.slider(
    "Illustrative transfer increase (%)",
    min_value=0,
    max_value=100,
    value=10,
    step=5,
)

# ============================================================
# 6. GENERATE FORECASTS
# ============================================================

with st.spinner("Generating forecasts..."):
    care_forecast = forecast(care, horizon, model_name)

    discharge_model = st.sidebar.selectbox(
        "Discharge forecast method",
        [
            "Moving Average",
            "Naive Persistence",
            "ARIMA",
            "Exponential Smoothing",
            "SARIMA",
        ],
        index=0,
    )

    discharge_forecast = basic_forecast(
        discharges, horizon, discharge_model
    )

future_dates = pd.date_range(
    start=df.index[-1] + pd.Timedelta(days=1),
    periods=horizon,
    freq="D",
)

forecast_df = pd.DataFrame(
    {
        "Date": future_dates,
        "Care Load Forecast": care_forecast,
        "Discharge Forecast": discharge_forecast,
    }
).set_index("Date")

# Approximate one-step residual band.
historical_errors = care.diff().dropna().tail(90)
error_low = float(historical_errors.quantile(0.025))
error_high = float(historical_errors.quantile(0.975))

forecast_df["Lower Band"] = np.maximum(
    0, care_forecast + error_low
)
forecast_df["Upper Band"] = np.maximum(
    0, care_forecast + error_high
)

# Illustrative sensitivity scenario, not a separately trained forecast.
# Assumption: extra transfers add to the active care load and are not offset
# by additional discharges. The effect accumulates across the forecast horizon.
recent_daily_transfers = max(float(transfers.tail(7).mean()), 0.0)
additional_transfers_per_day = (
    recent_daily_transfers * transfer_increase / 100.0
)
scenario_adjustments = additional_transfers_per_day * np.arange(1, horizon + 1)

forecast_df["Illustrative Scenario"] = np.maximum(
    0, care_forecast + scenario_adjustments
)
scenario_adjustment = float(scenario_adjustments[-1])

# ============================================================
# 7. HEADER AND KPI CARDS
# ============================================================

st.markdown(
    '<div class="dashboard-title">'
    'Predictive Forecasting of Care Load & Placement Demand</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'UAC Program | Historical trends, forecast demand, '
    'model performance and capacity planning</div>',
    unsafe_allow_html=True,
)

latest_care = float(care.iloc[-1])
latest_discharge = float(discharges.iloc[-1])
end_forecast = float(care_forecast[-1])
care_change = end_forecast - latest_care

k1, k2, k3, k4 = st.columns(4)

k1.metric("Latest HHS Care Load", f"{latest_care:,.0f}")
k2.metric(
    f"Care Load in {horizon} Days",
    f"{end_forecast:,.0f}",
    f"{care_change:+,.0f}",
)
k3.metric("Latest Daily Discharges", f"{latest_discharge:,.0f}")
k4.metric("Selected Forecast Model", model_name)

st.write("")

# ============================================================
# 8. CHART HELPERS
# ============================================================

CHART_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "responsive": True,
}

def style_chart(fig, height=330):
    fig.update_layout(
        height=height,
        margin=dict(l=15, r=15, t=45, b=20),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Arial", color="#53627A", size=11),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        hovermode="x unified",
    )

    fig.update_xaxes(
        showgrid=False,
        linecolor="#E6EBF2",
    )

    fig.update_yaxes(
        gridcolor="#E9EEF5",
        zerolinecolor="#E9EEF5",
    )

    return fig


def history_slice(series, selection):
    if selection == "All":
        return series

    return series.tail(int(selection))


# ============================================================
# 9. CARE LOAD AND DISCHARGE FORECASTS
# ============================================================

left, right = st.columns([1.65, 1])

with left:
    st.markdown(
        '<div class="section-title">Care Load Forecast</div>',
        unsafe_allow_html=True,
    )

    care_history = history_slice(care, history_days)

    fig_care = go.Figure()

    fig_care.add_trace(
        go.Scatter(
            x=care_history.index,
            y=care_history.values,
            name="Historical care load",
            mode="lines",
            line=dict(color="#2475E8", width=2.4),
        )
    )

    if show_interval:
        fig_care.add_trace(
            go.Scatter(
                x=future_dates,
                y=forecast_df["Upper Band"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig_care.add_trace(
            go.Scatter(
                x=future_dates,
                y=forecast_df["Lower Band"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(36,117,232,0.14)",
                name="Approximate uncertainty band",
            )
        )

    fig_care.add_trace(
        go.Scatter(
            x=future_dates,
            y=care_forecast,
            name="Forecast",
            mode="lines+markers",
            line=dict(color="#F29B38", width=3, dash="dash"),
            marker=dict(size=6),
        )
    )

    fig_care.add_vline(
        x=df.index[-1],
        line_dash="dot",
        line_color="#9AA6B8",
    )

    fig_care.update_layout(
        title=f"Next {horizon} Days",
        yaxis_title="Children in HHS Care",
    )

    style_chart(fig_care, 360)
    st.plotly_chart(
        fig_care,
        use_container_width=True,
        config=CHART_CONFIG,
    )

with right:
    st.markdown(
        '<div class="section-title">Discharge Demand</div>',
        unsafe_allow_html=True,
    )

    discharge_history = history_slice(discharges, history_days)

    fig_discharge = go.Figure()

    fig_discharge.add_trace(
        go.Scatter(
            x=discharge_history.index,
            y=discharge_history.values,
            name="Historical discharges",
            mode="lines",
            line=dict(color="#21A69A", width=2.2),
        )
    )

    fig_discharge.add_trace(
        go.Scatter(
            x=future_dates,
            y=discharge_forecast,
            name="Discharge forecast",
            mode="lines+markers",
            line=dict(color="#F29B38", width=2.5, dash="dash"),
        )
    )

    fig_discharge.update_layout(
        title=f"Next {horizon} Days",
        yaxis_title="Children discharged",
    )

    style_chart(fig_discharge, 360)
    st.plotly_chart(
        fig_discharge,
        use_container_width=True,
        config=CHART_CONFIG,
    )

# ============================================================
# 10. HISTORICAL FLOW ANALYSIS
# ============================================================

st.markdown("---")
st.markdown(
    '<div class="section-title">Intake and Exit Pressure</div>',
    unsafe_allow_html=True,
)

flow_left, flow_right = st.columns([1.5, 1])

with flow_left:
    flow_view = history_slice(
        df[[TRANSFER_COL, DISCHARGE_COL]], history_days
    )

    fig_flow = go.Figure()

    fig_flow.add_trace(
        go.Scatter(
            x=flow_view.index,
            y=flow_view[TRANSFER_COL],
            name="Transfers into HHS",
            mode="lines",
            line=dict(color="#2475E8", width=2),
        )
    )

    fig_flow.add_trace(
        go.Scatter(
            x=flow_view.index,
            y=flow_view[DISCHARGE_COL],
            name="HHS discharges",
            mode="lines",
            line=dict(color="#21A69A", width=2),
        )
    )

    fig_flow.update_layout(
        title="Daily Transfers vs. Discharges",
        yaxis_title="Children per day",
    )

    style_chart(fig_flow, 320)
    st.plotly_chart(
        fig_flow,
        use_container_width=True,
        config=CHART_CONFIG,
    )

with flow_right:
    pressure_view = history_slice(net_pressure, history_days)

    fig_pressure = go.Figure()

    fig_pressure.add_trace(
        go.Bar(
            x=pressure_view.index,
            y=pressure_view.values,
            name="Net pressure",
            marker_color="#7A8DA9",
        )
    )

    fig_pressure.add_hline(
        y=0,
        line_color="#64748B",
        line_dash="dot",
    )

    fig_pressure.update_layout(
        title="Net Pressure: Transfers − Discharges",
        yaxis_title="Net children per day",
    )

    style_chart(fig_pressure, 320)
    st.plotly_chart(
        fig_pressure,
        use_container_width=True,
        config=CHART_CONFIG,
    )

st.caption(
    "Net pressure is a flow indicator. It does not fully account for "
    "all changes in active HHS care load."
)

# ============================================================
# 11. MODEL COMPARISON
# ============================================================

st.markdown("---")
st.markdown(
    '<div class="section-title">Model Performance Comparison</div>',
    unsafe_allow_html=True,
)

if RESULTS_FILE.exists():
    results = pd.read_csv(RESULTS_FILE)

    required_metrics = {"Model", "MAE", "RMSE", "MAPE (%)"}

    if required_metrics.issubset(results.columns):
        results = results.dropna(subset=["MAE", "RMSE", "MAPE (%)"])
        results = results.sort_values("MAE").reset_index(drop=True)

        comparison_left, comparison_right = st.columns([1.05, 1.6])

        with comparison_left:
            st.dataframe(
                results.round(2),
                use_container_width=True,
                hide_index=True,
                height=min(350, 45 + len(results) * 36),
            )

            metric = st.selectbox(
                "Compare by metric",
                ["MAE", "RMSE", "MAPE (%)"],
            )

        with comparison_right:
            fig_models = go.Figure()

            fig_models.add_trace(
                go.Bar(
                    x=results["Model"],
                    y=results[metric],
                    marker_color=[
                        "#21A69A" if i == 0 else "#6D8FC7"
                        for i in range(len(results))
                    ],
                    text=results[metric].round(2),
                    textposition="outside",
                )
            )

            fig_models.update_layout(
                title=f"Model Comparison by {metric}",
                xaxis_title="Forecasting model",
                yaxis_title=metric,
            )

            fig_models.update_xaxes(tickangle=-25)

            style_chart(fig_models, 350)
            st.plotly_chart(
                fig_models,
                use_container_width=True,
                config=CHART_CONFIG,
            )

        st.caption(
            "Imported model metrics are preliminary. Complete leakage "
            "checks and validation before selecting a final model."
        )

    else:
        st.info(
            "The comparison CSV does not contain all expected metric columns."
        )
else:
    st.info(
        "Place model_comparison_results.csv inside the Data folder "
        "to display model performance."
    )

# ============================================================
# 12. SCENARIO COMPARISON
# ============================================================

st.markdown("---")
st.markdown(
    '<div class="section-title">Scenario Comparison</div>',
    unsafe_allow_html=True,
)

scenario_left, scenario_right = st.columns([1.6, 1])

with scenario_left:
    fig_scenario = go.Figure()

    fig_scenario.add_trace(
        go.Scatter(
            x=future_dates,
            y=care_forecast,
            name="Base forecast",
            mode="lines+markers",
            line=dict(color="#2475E8", width=3),
        )
    )

    fig_scenario.add_trace(
        go.Scatter(
            x=future_dates,
            y=forecast_df["Illustrative Scenario"],
            name=f"Illustrative +{transfer_increase}% transfer scenario",
            mode="lines+markers",
            line=dict(color="#F29B38", width=3, dash="dash"),
        )
    )

    fig_scenario.update_layout(
        title="Base Forecast vs. Illustrative Scenario",
        yaxis_title="Children in HHS Care",
    )

    style_chart(fig_scenario, 330)
    st.plotly_chart(
        fig_scenario,
        use_container_width=True,
        config=CHART_CONFIG,
    )

with scenario_right:
    st.markdown("#### Scenario summary")

    st.metric(
        "Base load at horizon",
        f"{care_forecast[-1]:,.0f}",
    )

    st.metric(
        "Illustrative scenario load",
        f"{forecast_df['Illustrative Scenario'].iloc[-1]:,.0f}",
    )

    st.metric(
        "Difference from base",
        f"{scenario_adjustment:+,.1f}",
    )

st.caption(
    "Illustrative sensitivity only: the selected percentage is applied to "
    "the recent 7-day average of daily transfers, and the additional load "
    "accumulates across the horizon. This assumes extra transfers are not "
    "offset by additional discharges; it is not a validated operational forecast."
)

# ============================================================
# 13. CAPACITY STRESS INDICATOR
# ============================================================

st.markdown("---")
st.markdown(
    '<div class="section-title">Capacity Stress Assessment</div>',
    unsafe_allow_html=True,
)

if capacity > 0:
    breach_days = int(np.sum(care_forecast > capacity))
    breach_rate = 100 * breach_days / horizon

    cap1, cap2, cap3 = st.columns(3)

    cap1.metric("Entered Capacity Threshold", f"{capacity:,.0f}")
    cap2.metric("Forecast Days Above Threshold", f"{breach_days}")
    cap3.metric("Forecast-Horizon Exceedance Rate", f"{breach_rate:.1f}%")

    st.caption(
        "This is based on forecast point estimates above the entered "
        "threshold. It is not a probability of a capacity breach."
    )
else:
    st.info(
        "Enter an approved HHS care capacity threshold in the sidebar "
        "to calculate forecast-horizon exceedance indicators."
    )

# ============================================================
# 14. FORECAST TABLE AND DOWNLOAD
# ============================================================

st.markdown("---")
st.markdown(
    '<div class="section-title">Forecast Details</div>',
    unsafe_allow_html=True,
)

display_forecast = forecast_df.copy().reset_index()
display_forecast = display_forecast.rename(columns={"index": "Date"})

st.dataframe(
    display_forecast.round(2),
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    label="Download Forecast Results (CSV)",
    data=display_forecast.to_csv(index=False).encode("utf-8"),
    file_name="uac_forecast_results.csv",
    mime="text/csv",
)

st.markdown(
    """
    <div class="info-note">
    Forecasts and uncertainty bands are analytical estimates based on
    historical data. They are not official operational guidance.
    Capacity-breach probability and surge lead time require additional
    validated methods and appropriate operational definitions.
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("UAC Care Demand Analytics | Predictive Forecasting Project")
