import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from openai import OpenAI
import os
import getpass
import statsmodels.api as sm

# -------------------------------
# OpenAI API Setup (v1.x)
# -------------------------------
# -------------------------------
# Secure API Key Handling
# -------------------------------
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ OpenAI API key not found in environment variables.")
        # Prompt securely without echoing input
        api_key = getpass.getpass("🔑 Please enter your OpenAI API key: ").strip()

        if not api_key:
            raise ValueError("❌ API key is required to run this script.")
        else:
            print("✅ API key received securely.")

    return OpenAI(api_key=api_key)

# Initialize OpenAI Client
client = get_openai_client()

# Example test call to verify it works:
try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a test AI assistant."},
                  {"role": "user", "content": "Say 'API Key Connected Successfully!'"}],
    )
    print("\n🤖 OpenAI Test Response:", response.choices[0].message.content)
except Exception as e:
    print("❌ API Call Failed:", str(e))

# -------------------------------
# Gauge (Tachometer Style)
# -------------------------------
def create_gauge(title, value, min_val, max_val, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 18}},
        gauge={
            "axis": {"range": [min_val, max_val]},
            "bar": {"color": color},
            "bgcolor": "white",
            "steps": [
                {"range": [min_val, max_val * 0.5], "color": "#e0f7fa"},
                {"range": [max_val * 0.8, max_val], "color": "#80deea"}
            ]
        }
    ))
    fig.update_layout(
        autosize=True,
        margin=dict(l=20, r=20, t=40, b=20),
        # ensure all text has room
    )
    return fig

# -------------------------------
# Agentic AI Prediction (OpenAI v1.x)
# -------------------------------
def agentic_ai_predict(species, stress, infection, wound, temp, pest, germ):
    prompt = f"""
    You are an AI veterinarian and econometrics expert. Based on these sensor readings:
    Species: {species}
    Stress Score: {stress} %
    Infection Risk: {infection} %
    Wound Area: {wound} cm²
    Temperature: {temp} °C
    Pest Risk: {pest} %
    Germ Load: {germ} %

    Provide:
    1. Key health observations.
    2. Urgent interventions.
    3. Long-term preventive recommendations.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an AI assistant for livestock health analytics and bioeconometric modeling."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
    )
    return response.choices[0].message.content

# -------------------------------
# Synthetic Dataset Generator
# -------------------------------
def generate_synthetic_data(n=100):
    species = np.random.choice(["Cattle", "Sheep", "Goat", "Pig", "Chicken"], size=n)
    return pd.DataFrame({
        "Species": species,
        "StressScore": np.random.uniform(20, 80, size=n),
        "InfectionRisk": np.random.uniform(5, 70, size=n),
        "WoundArea": np.random.uniform(0, 5, size=n),
        "Temperature": np.random.normal(38, 0.7, size=n),
        "PestRisk": np.random.uniform(5, 60, size=n),
        "GermLoad": np.random.uniform(5, 50, size=n)
    })
# -------------------------------
# Econometric OLS Model
# -------------------------------
def run_ols_model(df):
    X = df[["StressScore", "Temperature", "WoundArea", "PestRisk", "GermLoad"]]
    y = df["InfectionRisk"]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

# -------------------------------
# Streamlit UI Layout
# -------------------------------
st.set_page_config(page_title="AI Livestock Control Panel", layout="wide")
st.title("Smart Livestock Health Monitoring Dashboard")

uploaded_file = st.file_uploader("Upload Sensor Data (.csv or .xlsx)", type=["csv", "xlsx"])
use_uploaded = False
df = None

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    use_uploaded = True
    st.success("File uploaded successfully.")

if not use_uploaded:
    st.info("Upload Sensor Data")
    df = generate_synthetic_data()
    st.dataframe(df.head(10), height=250)
else:
    st.dataframe(df.head(10), height=250)

# Row Selection
selected_row = st.slider("Select Row for Analysis", 0, len(df) - 1, 0)
row = df.iloc[selected_row]

species = row["Species"]
stress = float(row["StressScore"])
infection = float(row["InfectionRisk"])
wound = float(row["WoundArea"])
temp = float(row["Temperature"])
pest = float(row["PestRisk"])
germ = float(row["GermLoad"])

st.markdown(f"### Selected Species: **{species}**")

# Gauges Display
g1, g2, g3 = st.columns(3)
with g1: st.plotly_chart(create_gauge("Stress Score (%)", stress, 0, 100, "orange"))
with g2: st.plotly_chart(create_gauge("Infection Risk (%)", infection, 0, 100, "red"))
with g3: st.plotly_chart(create_gauge("Wound Area (%)", wound * 10, 0, 100, "purple"))

g4, g5, g6 = st.columns(3)
with g4: st.plotly_chart(create_gauge("Temperature (°C)", (temp - 35) * (100/7), 0, 100, "blue"))
with g5: st.plotly_chart(create_gauge("Pest Risk (%)", pest, 0, 100, "green"))
with g6: st.plotly_chart(create_gauge("Germ Load (%)", germ, 0, 100, "brown"))

# AI Button
if st.button("Generate AI & Econometric Predictions"):
    # Create equal width responsive columns
    col1, col2 = st.columns([1, 1], gap="large")

    # ----------- LEFT COLUMN: Agentic AI Diagnosis -----------
    with col1:
        st.subheader("Agentic AI and Econometric Diagnosis")
        with st.spinner("Generating results..."):
            ai_diagnosis = agentic_ai_predict(species, stress, infection, wound, temp, pest, germ)

        # Styled container for AI output
        st.markdown(
            f"""
            <div style="
                background-color: white;
                color: black;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #ddd;
                font-size: 16px;
                line-height: 1.5;
                overflow-y: auto;
                max-height: 400px;
            ">
                {ai_diagnosis}
            </div>
            """,
            unsafe_allow_html=True
        )

    # ----------- RIGHT COLUMN: OLS Econometric Prediction -----------
    with col2:
        st.subheader("Econometric Prediction")

        # Train model
        model = run_ols_model(df)

        # Prepare input row
        input_df = pd.DataFrame([[stress, temp, wound, pest, germ]],
                                columns=["StressScore", "Temperature", "WoundArea", "PestRisk", "GermLoad"])
        input_df = sm.add_constant(input_df, has_constant='add')
        input_df = input_df[model.model.exog_names]  # Align

        # Prediction with confidence intervals
        prediction = model.get_prediction(input_df)
        pred_summary = prediction.summary_frame(alpha=0.05)

        predicted_infection = pred_summary["mean"].iloc[0]
        lower_ci = pred_summary["mean_ci_lower"].iloc[0]
        upper_ci = pred_summary["mean_ci_upper"].iloc[0]

        # Display prediction
        st.metric("Predicted Infection Risk (%)", f"{predicted_infection:.2f}")
        st.caption(f"95% Confidence Interval: **{lower_ci:.2f}% – {upper_ci:.2f}%**")

        # Residuals vs Fitted Plot
        residuals = model.resid
        fitted = model.fittedvalues
        resid_fig = go.Figure()
        resid_fig.add_trace(go.Scatter(x=fitted, y=residuals, mode='markers', marker=dict(color='teal')))
        resid_fig.add_hline(y=0, line_dash='dash', line_color='red')
        resid_fig.update_layout(
            title="Residuals vs Fitted",
            xaxis_title="Fitted Values",
            yaxis_title="Residuals",
            height=300,
            margin=dict(l=10, r=10, t=40, b=20)
        )
        st.plotly_chart(resid_fig, use_container_width=True)

        with st.expander("View OLS Model Summary"):
            st.code(model.summary().as_text())


# ──────────────────────────────────────────────
# FOOTER ANNOTATIONS
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    .footer {
        position: Draggable;
        left: 0;
        bottom: 0;
        width: 100%;
        background: rgba(0, 0, 0, 0.85);
        text-align: center;
        padding: 10px 0;
    }
    </style>
    <div class='footer'>
        <h4 style='color: #2ecc71; margin: 0;'>Developed and Designed by <b>Jit</b></h4>
        <p style='font-size: 14px; color: #bdc3c7; margin: 3px 0;'>
            <i> LiveStock Monitoring – AI & Econometric Intelligence for Next-Gen Agriculture</i>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
