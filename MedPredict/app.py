import os
from pathlib import Path

import joblib
import streamlit as st
import plotly.express as px

# Resolve paths relative to this app.py file. This is important on Streamlit Cloud,
# where the working directory is the repository root.
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

from src.data import load_train, add_rul, add_features


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="MedPredict",
    page_icon="🏥",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🏥 MedPredict")

st.subheader(
    "Predictive Maintenance & Equipment Monitoring"
)

st.caption(
    "Healthcare-oriented predictive maintenance prototype "
    "using NASA C-MAPSS simulated sensor data."
)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

try:
    bundle = joblib.load(
        BASE_DIR / "models" / "medpredict.joblib"
    )

except FileNotFoundError:

    st.error(
        "Model not found.\n\n"
        "Please run:\n"
        "1. python download_data.py\n"
        "2. python train.py"
    )

    st.stop()


# =========================================================
# LOAD DATA
# =========================================================

df = load_train()

df = add_rul(df)

df = add_features(df)

units = sorted(
    df["unit"].unique()
)


# =========================================================
# EQUIPMENT INFORMATION
# =========================================================
# These are simulated portfolio metadata.
# They are NOT real GE equipment records.

equipment_info = {

    1: {
        "name": "MRI Scanner",
        "model": "MRI Imaging System",
        "location": "Bangalore - Imaging Center",
    },

    2: {
        "name": "CT Scanner",
        "model": "CT Imaging System",
        "location": "Bangalore - Radiology",
    },

    3: {
        "name": "Ultrasound Machine",
        "model": "Ultrasound Imaging System",
        "location": "Bangalore - Diagnostics",
    },

    4: {
        "name": "Digital X-Ray System",
        "model": "Digital Radiography System",
        "location": "Bangalore - Radiology",
    },

    5: {
        "name": "Patient Monitor",
        "model": "Multi-Parameter Patient Monitor",
        "location": "Bangalore - ICU",
    },
}


# =========================================================
# SENSOR DISPLAY NAMES
# =========================================================
# C-MAPSS uses anonymized sensor columns.
# These names are intentionally generic/technical.

sensor_names = {

    "s1": "Temperature Sensor",
    "s2": "Pressure Sensor",
    "s3": "Vibration Sensor",
    "s4": "Speed Sensor",
    "s5": "Flow Sensor",
    "s6": "Power Sensor",
    "s7": "Motor Current Sensor",
    "s8": "Voltage Sensor",
    "s9": "Frequency Sensor",
    "s10": "Torque Sensor",
    "s11": "Load Sensor",
    "s12": "Coolant Temperature Sensor",
    "s13": "Bearing Sensor",
    "s14": "Fan Speed Sensor",
    "s15": "Exhaust Sensor",
    "s16": "Pressure Monitoring Sensor",
    "s17": "Thermal Sensor",
    "s18": "Performance Sensor",
    "s19": "Flow Monitoring Sensor",
    "s20": "Power Monitoring Sensor",
    "s21": "System Health Sensor",
}


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Equipment Selection")


display_names = []

for unit in units:

    equipment = equipment_info.get(
        unit,
        {
            "name": f"Equipment {unit}",
            "model": "Unknown",
            "location": "Unknown"
        }
    )

    display_names.append(
        equipment["name"]
    )


selected_name = st.sidebar.selectbox(
    "Select Equipment",
    display_names
)


# Convert selected equipment name back to dataset ID

name_to_id = {}

for unit in units:

    equipment = equipment_info.get(
        unit,
        {
            "name": f"Equipment {unit}"
        }
    )

    name_to_id[
        equipment["name"]
    ] = unit


unit = name_to_id[selected_name]


# =========================================================
# SELECT EQUIPMENT DATA
# =========================================================

history = df[
    df["unit"] == unit
].copy()


latest = history.iloc[-1]


# =========================================================
# RUL PREDICTION
# =========================================================

X = latest[
    bundle["features"]
].to_frame().T


rul = max(
    0,
    float(
        bundle["model"].predict(X)[0]
    )
)


# =========================================================
# STATUS & MAINTENANCE PRIORITY
# =========================================================

if rul <= 10:

    status = "Critical"

    maintenance_priority = (
        "Immediate Maintenance"
    )

elif rul <= 30:

    status = "High Risk"

    maintenance_priority = (
        "High Priority"
    )

elif rul <= 60:

    status = "Monitoring"

    maintenance_priority = (
        "Medium Priority"
    )

else:

    status = "Healthy"

    maintenance_priority = (
        "Low Priority"
    )


# =========================================================
# EQUIPMENT INFORMATION
# =========================================================

info = equipment_info.get(
    unit,
    {
        "name": f"Equipment {unit}",
        "model": "Unknown",
        "location": "Unknown"
    }
)


# =========================================================
# EQUIPMENT OVERVIEW
# =========================================================

st.markdown(
    "## Equipment Overview"
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Equipment Name",
        info["name"]
    )

    st.metric(
        "Model",
        info["model"]
    )


with col2:

    st.metric(
        "Location",
        info["location"]
    )

    st.metric(
        "Current Operating Cycle",
        int(latest["cycle"])
    )


with col3:

    st.metric(
        "Remaining Useful Life",
        f"{rul:.1f} cycles"
    )

    st.metric(
        "Status",
        status
    )


# =========================================================
# MAINTENANCE ASSESSMENT
# =========================================================

st.markdown(
    "## Maintenance Assessment"
)


col1, col2 = st.columns(2)


with col1:

    st.info(
        f"**Equipment:** {info['name']}\n\n"
        f"**Model:** {info['model']}\n\n"
        f"**Location:** {info['location']}\n\n"
        f"**Current Cycle:** {int(latest['cycle'])}"
    )


with col2:

    if maintenance_priority == "Immediate Maintenance":

        st.error(
            f"🚨 **{maintenance_priority}**\n\n"
            f"Predicted RUL: {rul:.1f} cycles"
        )

    elif maintenance_priority == "High Priority":

        st.warning(
            f"⚠️ **{maintenance_priority}**\n\n"
            f"Predicted RUL: {rul:.1f} cycles"
        )

    elif maintenance_priority == "Medium Priority":

        st.warning(
            f"🔎 **{maintenance_priority}**\n\n"
            f"Predicted RUL: {rul:.1f} cycles"
        )

    else:

        st.success(
            f"✅ **{maintenance_priority}**\n\n"
            f"Predicted RUL: {rul:.1f} cycles"
        )


# =========================================================
# SENSOR MONITORING
# =========================================================

st.markdown(
    "## Sensor Monitoring"
)


# Create dropdown with friendly names

selected_sensor_name = st.selectbox(
    "Select Operating Parameter",
    list(sensor_names.values())
)


# Convert friendly name back to actual dataset column

sensor = next(
    key
    for key, value in sensor_names.items()
    if value == selected_sensor_name
)


# =========================================================
# SENSOR GRAPH
# =========================================================

fig = px.line(
    history,
    x="cycle",
    y=sensor,
    title=(
        f"{selected_sensor_name} Trend - "
        f"{info['name']}"
    ),
    labels={
        "cycle": "Operating Cycle",
        sensor: selected_sensor_name
    }
)


st.plotly_chart(
    fig,
    width='stretch'
)


# =========================================================
# CURRENT SENSOR VALUE
# =========================================================

current_sensor_value = latest[sensor]

average_sensor_value = history[sensor].mean()


col1, col2 = st.columns(2)


with col1:

    st.metric(
        "Current Reading",
        f"{current_sensor_value:.2f}"
    )


with col2:

    st.metric(
        "Average Reading",
        f"{average_sensor_value:.2f}"
    )


# =========================================================
# RECENT SENSOR READINGS
# =========================================================

st.markdown(
    "## Recent Sensor Readings"
)


# Show only the selected sensor and cycle

recent_data = history[
    ["cycle", sensor]
].tail(10).copy()


recent_data.columns = [
    "Operating Cycle",
    selected_sensor_name
]


st.dataframe(
    recent_data,
    width='stretch',
    hide_index=True
)


# =========================================================
# PROJECT SUMMARY
# =========================================================

st.markdown(
    "## Prediction Summary"
)


st.write(
    f"""
    **Equipment:** {info['name']}

    **Model:** {info['model']}

    **Location:** {info['location']}

    **Predicted Remaining Useful Life:** {rul:.1f} cycles

    **Current Status:** {status}

    **Maintenance Priority:** {maintenance_priority}
    """
)


# =========================================================
# DISCLAIMER
# =========================================================

st.divider()

st.caption(
    "⚠️ Portfolio prototype only. Equipment names, models and "
    "locations are simulated metadata. The underlying C-MAPSS "
    "dataset contains simulated turbofan-engine sensor data and "
    "should not be interpreted as real medical-device data or "
    "used for clinical or maintenance decisions."
)