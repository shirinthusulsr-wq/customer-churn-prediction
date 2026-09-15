import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# LOAD MODEL COMPONENTS
# ============================================================

model = joblib.load("Model/logistic_churn_model.pkl")
scaler = joblib.load("Model/scaler.pkl")
business_parameters = joblib.load("Model/business_parameters.pkl")

# Use the exact feature structure remembered by the scaler.
scaler_features = scaler.feature_names_in_.tolist()

churn_threshold = business_parameters["churn_threshold"]
cltv_median = business_parameters["cltv_median"]


# ============================================================
# HEADER
# ============================================================

st.title("Customer Churn Prediction & Retention Analytics")

st.write(
    "Enter customer information below to estimate churn risk "
    "and determine retention priority."
)

st.divider()


# ============================================================
# CUSTOMER INPUTS
# ============================================================

st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)


# ============================================================
# COLUMN 1 - CUSTOMER PROFILE
# ============================================================

with col1:
    st.markdown("### Customer Profile")

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=72,
        value=12,
        step=1
    )

    cltv = st.number_input(
        "Customer Lifetime Value (CLTV)",
        min_value=0.0,
        value=4500.0,
        step=100.0
    )


# ============================================================
# COLUMN 2 - SERVICES
# ============================================================

with col2:
    st.markdown("### Services")

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )


# ============================================================
# COLUMN 3 - ACCOUNT AND BILLING
# ============================================================

with col3:
    st.markdown("### Account & Billing")

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=1.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1000.0,
        step=50.0
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "Predict Churn Risk",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION LOGIC
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # Create a blank row using EXACT scaler feature names
    # --------------------------------------------------------

    customer_encoded = pd.DataFrame(
        0.0,
        index=[0],
        columns=scaler_features
    )

    # --------------------------------------------------------
    # NUMERICAL FEATURES
    # --------------------------------------------------------

    numerical_values = {
        "Senior Citizen": 1 if senior_citizen == "Yes" else 0,
        "Tenure Months": tenure,
        "Monthly Charges": monthly_charges,
        "Total Charges": total_charges,
        "CLTV": cltv
    }

    for feature_name, feature_value in numerical_values.items():
        if feature_name in customer_encoded.columns:
            customer_encoded.loc[0, feature_name] = feature_value

    # --------------------------------------------------------
    # CATEGORICAL FEATURES
    # --------------------------------------------------------

    categorical_values = {
        "Gender": gender,
        "Partner": partner,
        "Dependents": dependents,
        "Phone Service": phone_service,
        "Multiple Lines": multiple_lines,
        "Internet Service": internet_service,
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection": device_protection,
        "Tech Support": tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies,
        "Contract": contract,
        "Paperless Billing": paperless_billing,
        "Payment Method": payment_method
    }

    for feature_name, feature_value in categorical_values.items():
        encoded_name = f"{feature_name}_{feature_value}"

        if encoded_name in customer_encoded.columns:
            customer_encoded.loc[0, encoded_name] = 1.0

    # --------------------------------------------------------
    # GUARANTEE EXACT FEATURE ORDER
    # --------------------------------------------------------

    customer_encoded = customer_encoded.loc[
        :,
        scaler_features
    ]

    # --------------------------------------------------------
    # SCALE CUSTOMER DATA
    # --------------------------------------------------------

    customer_scaled = scaler.transform(
        customer_encoded
    )

    # --------------------------------------------------------
    # PREDICT CHURN PROBABILITY
    # --------------------------------------------------------

    churn_probability = model.predict_proba(
        customer_scaled
    )[0][1]

    # --------------------------------------------------------
    # DETERMINE CHURN CLASS
    # --------------------------------------------------------

    predicted_churn = (
    churn_probability >= churn_threshold
    )

    # --------------------------------------------------------
    # DETERMINE RISK LEVEL
    # --------------------------------------------------------

    if churn_probability < 0.30:
        risk_level = "Low"

    elif churn_probability < 0.40:
        risk_level = "Moderate"

    elif churn_probability < 0.60:
        risk_level = "High"

    else:
        risk_level = "Very High"

    # --------------------------------------------------------
    # DETERMINE CUSTOMER VALUE
    # --------------------------------------------------------

    if cltv >= cltv_median:
        customer_value = "High Value"

    else:
        customer_value = "Lower Value"

    # --------------------------------------------------------
    # DETERMINE RETENTION PRIORITY
    # --------------------------------------------------------

    if predicted_churn and customer_value == "High Value":
        retention_priority = "Priority Retention"

    elif predicted_churn:
        retention_priority = "Retention Candidate"

    elif customer_value == "High Value":
        retention_priority = "Relationship Maintenance"

    else:
        retention_priority = "Lower Priority"

    # ========================================================
    # RESULTS
    # ========================================================

    st.subheader("Prediction Results")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric(
            "Churn Probability",
            f"{churn_probability:.1%}"
        )

    with result_col2:
        st.metric(
            "Risk Level",
            risk_level
        )

    with result_col3:
        st.metric(
            "Customer Value",
            customer_value
        )

    # --------------------------------------------------------
    # RISK BAR
    # --------------------------------------------------------

    st.markdown("### Churn Risk")

    st.progress(
        float(churn_probability)
    )

    st.caption(
        f"Model classification threshold: "
        f"{churn_threshold:.0%}"
    )

    # --------------------------------------------------------
    # RETENTION RECOMMENDATION
    # --------------------------------------------------------

    st.markdown("### Retention Recommendation")

    if retention_priority == "Priority Retention":
        st.error(
            "🔴 PRIORITY RETENTION — This customer has elevated "
            "predicted churn risk and above-median customer value. "
            "Consider proactive retention intervention."
        )

    elif retention_priority == "Retention Candidate":
        st.warning(
            "🟠 RETENTION CANDIDATE — This customer has elevated "
            "predicted churn risk. Consider an appropriate "
            "retention intervention."
        )

    elif retention_priority == "Relationship Maintenance":
        st.info(
            "🔵 RELATIONSHIP MAINTENANCE — This customer has "
            "relatively low predicted churn risk but high customer "
            "value. Continue relationship management and engagement."
        )

    else:
        st.success(
            "🟢 LOWER PRIORITY — This customer currently has "
            "relatively low predicted churn risk and lower "
            "customer value."
        )

    # --------------------------------------------------------
    # DETAILS
    # --------------------------------------------------------

    with st.expander("View Prediction Details"):
        st.write(
            f"**Predicted churn probability:** "
            f"{churn_probability:.2%}"
        )

        st.write(
            f"**Classification threshold:** "
            f"{churn_threshold:.0%}"
        )

        st.write(
            f"**Risk category:** "
            f"{risk_level}"
        )

        st.write(
            f"**Customer CLTV:** "
            f"{cltv:,.0f}"
        )

        st.write(
            f"**High-value CLTV threshold:** "
            f"{cltv_median:,.0f}"
        )

        st.write(
            f"**Customer value category:** "
            f"{customer_value}"
        )

        st.write(
            f"**Retention priority:** "
            f"{retention_priority}"
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

with st.expander("About the Model"):
    st.markdown(
        """
        **Model:** Logistic Regression

        **ROC-AUC:** 0.849

        **Operating classification threshold:** 0.40

        The Logistic Regression model was compared with a Random
        Forest classifier and provided stronger baseline performance
        across most evaluation metrics.

        The classification threshold was reduced from 0.50 to 0.40
        to increase recall and identify a larger proportion of actual
        churners.

        Customer churn probability is combined with Customer Lifetime
        Value (CLTV) to create a retention-priority framework.

        Predictions represent statistical associations and should not
        be interpreted as causal relationships.
        """
    )