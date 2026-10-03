from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.exceptions import NotFittedError


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "loan_approval_model.pkl"


st.set_page_config(
    page_title="LoanSure | Loan Approval Prediction",
    page_icon=":material/account_balance:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
        :root {
            --bg: #f3f7f8;
            --surface: #ffffff;
            --ink: #172033;
            --muted: #687385;
            --line: #dce3ec;
            --accent: #0f766e;
            --accent-strong: #115e59;
            --success-bg: #ecfdf5;
            --success-line: #7dd3a8;
            --danger-bg: #fff1f2;
            --danger-line: #fda4af;
        }

        .stApp {
            background:
                radial-gradient(ellipse at 12% 8%, rgba(20, 184, 166, 0.13), transparent 34%),
                radial-gradient(ellipse at 88% 22%, rgba(59, 130, 246, 0.09), transparent 30%),
                var(--bg);
            color: var(--ink);
        }

        .stApp::before {
            content: "";
            position: fixed;
            inset: -20%;
            z-index: 0;
            pointer-events: none;
            background: radial-gradient(circle, rgba(45, 212, 191, 0.08) 0, transparent 42%);
            filter: blur(24px);
            animation: ambient-drift 18s ease-in-out infinite alternate;
        }

        @keyframes ambient-drift {
            from { transform: translate3d(-2%, -1%, 0) scale(0.96); }
            to { transform: translate3d(3%, 2%, 0) scale(1.06); }
        }

        .block-container {
            position: relative;
            z-index: 1;
            max-width: 1180px;
            padding: 2rem 2rem 3rem;
        }

        #MainMenu, footer, header {
            visibility: hidden;
        }

        .app-header {
            display: flex;
            justify-content: space-between;
            gap: 1.5rem;
            align-items: flex-end;
            padding: 1.4rem 0 1.35rem;
            border-bottom: 1px solid var(--line);
            margin-bottom: 1.4rem;
        }

        .brand-kicker {
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }

        .brand-title {
            color: var(--ink);
            font-size: 2.35rem;
            line-height: 1.05;
            font-weight: 850;
            margin: 0;
            letter-spacing: -0.045em;
        }

        .brand-subtitle {
            color: var(--muted);
            font-size: 1rem;
            margin-top: 0.55rem;
            max-width: 680px;
        }

        .status-panel {
            min-width: 240px;
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 0.85rem 1rem;
            box-shadow: 0 14px 36px rgba(23, 32, 51, 0.08);
            backdrop-filter: blur(12px);
        }

        .status-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
        }

        .status-value {
            color: var(--ink);
            font-size: 1rem;
            font-weight: 800;
            margin-top: 0.2rem;
        }

        .section-heading {
            color: var(--ink);
            font-size: 1.12rem;
            font-weight: 820;
            margin: 1.55rem 0 0.25rem;
            letter-spacing: -0.02em;
        }

        .section-copy {
            color: var(--muted);
            font-size: 0.91rem;
            margin: 0 0 0.75rem;
        }

        div[data-testid="stForm"] {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(220, 227, 236, 0.9);
            border-radius: 20px;
            padding: 1.35rem 1.5rem 1.5rem;
            box-shadow: 0 22px 60px rgba(23, 32, 51, 0.08);
            backdrop-filter: blur(16px);
            animation: rise-in 650ms cubic-bezier(.2,.75,.25,1) both;
        }

        @keyframes rise-in {
            from { opacity: 0; transform: translateY(14px); }
            to { opacity: 1; transform: translateY(0); }
        }

        label, .stNumberInput label, .stSelectbox label {
            color: #263244 !important;
            font-size: 0.86rem !important;
            font-weight: 700 !important;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div {
            border-radius: 10px;
            border-color: #cfd8e3;
            min-height: 46px;
            background: rgba(255, 255, 255, 0.9);
            transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
        }

        div[data-baseweb="input"]:focus-within > div,
        div[data-baseweb="select"]:focus-within > div {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.12);
            transform: translateY(-1px);
        }

        .stButton > button,
        .stFormSubmitButton > button {
            width: 100%;
            min-height: 48px;
            border-radius: 11px;
            border: 1px solid var(--accent);
            background: var(--accent);
            color: #ffffff;
            font-size: 0.98rem;
            font-weight: 800;
            box-shadow: 0 10px 22px rgba(15, 118, 110, 0.2);
            transition: transform 160ms ease, box-shadow 160ms ease, background 160ms ease;
        }

        .stButton > button:hover,
        .stFormSubmitButton > button:hover {
            border-color: var(--accent-strong);
            background: var(--accent-strong);
            color: #ffffff;
            transform: translateY(-1px);
            box-shadow: 0 14px 28px rgba(15, 118, 110, 0.26);
        }

        .result-card {
            border-radius: 18px;
            padding: 1.35rem;
            border: 1px solid;
            margin-top: 1.3rem;
            box-shadow: 0 18px 44px rgba(23, 32, 51, 0.08);
            animation: rise-in 500ms ease both;
        }

        .result-approved {
            background: var(--success-bg);
            border-color: var(--success-line);
        }

        .result-rejected {
            background: var(--danger-bg);
            border-color: var(--danger-line);
        }

        .result-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
        }

        .result-title {
            color: var(--ink);
            font-size: 1.75rem;
            font-weight: 850;
            margin-top: 0.15rem;
        }

        .result-copy {
            color: #445066;
            margin-top: 0.45rem;
            font-size: 0.96rem;
        }

        .metric-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 1rem 0 0.2rem;
        }

        .metric-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 15px;
            padding: 0.9rem 1rem;
            box-shadow: 0 8px 24px rgba(23, 32, 51, 0.045);
            transition: transform 180ms ease, box-shadow 180ms ease;
        }

        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 14px 30px rgba(23, 32, 51, 0.09);
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 750;
        }

        .metric-value {
            color: var(--ink);
            font-size: 1.35rem;
            font-weight: 850;
            margin-top: 0.2rem;
        }

        .disclaimer {
            color: var(--muted);
            text-align: center;
            font-size: 0.82rem;
            margin-top: 2rem;
        }

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                scroll-behavior: auto !important;
                transition-duration: 0.01ms !important;
            }
        }

        @media (max-width: 760px) {
            .block-container {
                padding: 1.2rem 1rem 2.2rem;
            }

            .app-header {
                display: block;
            }

            .brand-title {
                font-size: 2rem;
            }

            .status-panel {
                min-width: 0;
                margin-top: 1rem;
            }

            .metric-strip {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def section(title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="section-heading">{title}</div>
        <div class="section-copy">{copy}</div>
        """,
        unsafe_allow_html=True,
    )


def money(value: float) -> str:
    return f"${value:,.0f}"


try:
    model = load_model()
    model_status = "Model loaded"
except Exception as exc:
    model = None
    model_status = "Model unavailable"
    model_error = exc


st.markdown(
    f"""
    <div class="app-header">
        <div>
            <div class="brand-kicker">AI loan assessment</div>
            <h1 class="brand-title">LoanSure</h1>
            <div class="brand-subtitle">
                Evaluate applicant, financial, and loan details with a focused prediction workflow.
            </div>
        </div>
        <div class="status-panel">
            <div class="status-label">System status</div>
            <div class="status-value">{model_status}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None:
    st.error("The prediction model could not be loaded. Check that loan_approval_model.pkl exists in streamlit_app.")
    st.caption(f"Technical detail: {model_error}")
    st.stop()


with st.form("loan_prediction_form"):
    section(
        "Applicant Profile",
        "Basic identity, household, education, and credit information.",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 18, 100, 30, help="Applicant's age in completed years. Enter a value from 18 to 100.")
    with col2:
        dependents = st.number_input("Dependents", 0, 15, 0, help="People who rely on the applicant financially, such as children or other supported family members.")
    with col3:
        gender = st.selectbox("Gender", ["Male", "Female"], help="Select the applicant's gender category used by this model.")

    col1, col2, col3 = st.columns(3)
    with col1:
        marital_status = st.selectbox("Marital status", ["Single", "Married"], help="Choose the applicant's current marital status.")
    with col2:
        education_level = st.selectbox("Education level", ["Graduate", "Not Graduate"], help="Graduate means the applicant has completed a degree or equivalent higher education.")
    with col3:
        credit_score = st.number_input("Credit score", 300, 850, 650, help="Applicant's current credit bureau score. Use the score shown by your credit provider (300-850).")

    section(
        "Employment",
        "Employment type, employer category, and current loan obligations.",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        employment_status = st.selectbox(
            "Employment status",
            ["Salaried", "Self Employed", "Unemployed"],
            help="Select salaried for regular payroll employment, self employed for business or freelance income, or unemployed if currently without work.",
        )
    with col2:
        employer_category = st.selectbox(
            "Employer category",
            ["Government", "MNC", "Private", "Unemployed"],
            help="Choose the employer group used in the training data: Government, MNC, Private, or Unemployed.",
        )
    with col3:
        existing_loans = st.number_input("Existing loans", 0, 20, 0, help="Count of other active loans the applicant currently repays. Do not include this new request.")

    section(
        "Financial Position",
        "Income, assets, requested borrowing, and debt-to-income details.",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        applicant_income = st.number_input(
            "Applicant income",
            min_value=0.0,
            value=50000.0,
            step=1000.0,
            help="Applicant's gross income. Use the currency unit and time period expected by the training data, and enter 0 if there is no personal income.",
        )
    with col2:
        coapplicant_income = st.number_input(
            "Coapplicant income",
            min_value=0.0,
            value=0.0,
            step=1000.0,
            help="Coapplicant's gross income using the same currency unit and time period as applicant income. Use 0 when there is no coapplicant income.",
        )
    with col3:
        savings = st.number_input(
            "Savings",
            min_value=0.0,
            value=100000.0,
            step=5000.0,
            help="Total liquid savings available to the applicant, in the same currency unit as the other financial values.",
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        collateral_value = st.number_input(
            "Collateral value",
            min_value=0.0,
            value=200000.0,
            step=5000.0,
            help="Estimated current value of assets pledged to secure the loan, in the same currency unit as the other financial values. Enter 0 if unsecured.",
        )
    with col2:
        loan_amount = st.number_input(
            "Loan amount",
            min_value=0.0,
            value=200000.0,
            step=5000.0,
            help="Amount the applicant wants to borrow, in the same currency unit as the other financial values.",
        )
    with col3:
        dti_ratio = st.number_input(
            "Debt-to-income ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.01,
            format="%.2f",
            help="Monthly debt payments divided by monthly gross income as a decimal from 0 to 1. For example, enter 0.30 for 30%.",
        )

    section(
        "Loan Request",
        "Loan term, purpose, and property area for the requested facility.",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        loan_term = st.number_input("Loan term (months)", 1, 480, 60, help="Requested repayment duration in months. For example, 60 means five years.")
    with col2:
        loan_purpose = st.selectbox(
            "Loan purpose",
            ["Car", "Education", "Home", "Personal"],
            help="Select the main reason for borrowing: vehicle purchase, education costs, home purchase or improvement, or personal expenses.",
        )
    with col3:
        property_area = st.selectbox(
            "Property area",
            ["Semiurban", "Urban", "Rural"],
            help="Choose the area type for the property associated with this application. Use the closest matching category.",
        )

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
    predict = st.form_submit_button("Predict loan approval")


validation_errors = []
if predict:
    if applicant_income + coapplicant_income <= 0:
        validation_errors.append("Enter income greater than 0 for the applicant or coapplicant.")
    if loan_amount <= 0:
        validation_errors.append("Loan amount must be greater than 0.")

    for message in validation_errors:
        st.error(message)

if predict and not validation_errors:
    education_encoded = 1 if education_level == "Graduate" else 0

    input_data = pd.DataFrame(
        {
            "Applicant_Income": [applicant_income],
            "Coapplicant_Income": [coapplicant_income],
            "Age": [age],
            "Dependents": [dependents],
            "Credit_Score": [credit_score],
            "Existing_Loans": [existing_loans],
            "DTI_Ratio": [dti_ratio],
            "Savings": [savings],
            "Collateral_Value": [collateral_value],
            "Loan_Amount": [loan_amount],
            "Loan_Term": [loan_term],
            "Education_Level": [education_encoded],
            "Employment_Status_Salaried": [1 if employment_status == "Salaried" else 0],
            "Employment_Status_Self-employed": [1 if employment_status == "Self Employed" else 0],
            "Employment_Status_Unemployed": [1 if employment_status == "Unemployed" else 0],
            "Marital_Status_Single": [1 if marital_status == "Single" else 0],
            "Loan_Purpose_Car": [1 if loan_purpose == "Car" else 0],
            "Loan_Purpose_Education": [1 if loan_purpose == "Education" else 0],
            "Loan_Purpose_Home": [1 if loan_purpose == "Home" else 0],
            "Loan_Purpose_Personal": [1 if loan_purpose == "Personal" else 0],
            "Property_Area_Semiurban": [1 if property_area == "Semiurban" else 0],
            "Property_Area_Urban": [1 if property_area == "Urban" else 0],
            "Gender_Male": [1 if gender == "Male" else 0],
            "Employer_Category_Government": [1 if employer_category == "Government" else 0],
            "Employer_Category_MNC": [1 if employer_category == "MNC" else 0],
            "Employer_Category_Private": [1 if employer_category == "Private" else 0],
            "Employer_Category_Unemployed": [1 if employer_category == "Unemployed" else 0],
        }
    )

    # Match the feature engineering and column order used by the training notebook.
    input_data["DTI_Ratio_Squre"] = input_data["DTI_Ratio"] ** 2
    input_data["Credit_Score_Square"] = input_data["Credit_Score"] ** 2
    input_data["Applicant_Income_Log"] = np.log1p(input_data["Applicant_Income"])

    feature_order = [
        "Coapplicant_Income",
        "Age",
        "Dependents",
        "Existing_Loans",
        "Savings",
        "Collateral_Value",
        "Loan_Amount",
        "Loan_Term",
        "Education_Level",
        "Employment_Status_Salaried",
        "Employment_Status_Self-employed",
        "Employment_Status_Unemployed",
        "Marital_Status_Single",
        "Loan_Purpose_Car",
        "Loan_Purpose_Education",
        "Loan_Purpose_Home",
        "Loan_Purpose_Personal",
        "Property_Area_Semiurban",
        "Property_Area_Urban",
        "Gender_Male",
        "Employer_Category_Government",
        "Employer_Category_MNC",
        "Employer_Category_Private",
        "Employer_Category_Unemployed",
        "DTI_Ratio_Squre",
        "Credit_Score_Square",
        "Applicant_Income_Log",
    ]

    input_data = input_data[feature_order]

    try:
        prediction = model.predict(input_data)
        probability = None

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_data)[0][1] * 100

        approved = prediction[0] == 1
        result_class = "result-approved" if approved else "result-rejected"
        result_title = "Loan Approved" if approved else "Loan Not Approved"
        result_copy = (
            "The applicant profile is predicted to meet the approval pattern learned by the model."
            if approved
            else "The applicant profile is predicted to fall outside the model's approval pattern."
        )

        st.markdown(
            f"""
            <div class="result-card {result_class}">
                <div class="result-label">Prediction result</div>
                <div class="result-title">{result_title}</div>
                <div class="result-copy">{result_copy}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        probability_display = f"{probability:.2f}%" if probability is not None else "N/A"
        st.markdown(
            f"""
            <div class="metric-strip">
                <div class="metric-card">
                    <div class="metric-label">Approval probability</div>
                    <div class="metric-value">{probability_display}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Requested amount</div>
                    <div class="metric-value">{money(loan_amount)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Credit score</div>
                    <div class="metric-value">{credit_score}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Review model input"):
            st.dataframe(input_data, use_container_width=True, hide_index=True)

    except NotFittedError as exc:
        st.error(
            "The prediction model needs to be trained. Please restart the app after "
            "the model artifact has been refreshed."
        )
        st.caption(f"Technical detail: {exc}")
    except Exception as exc:
        st.error("We couldn't make a prediction from these details. Please review the inputs and try again.")
        st.caption(f"Technical detail: {exc}")


st.markdown(
    """
    <div class="disclaimer">
        LoanSure is a machine-learning demonstration and should not be used as the only basis for credit decisions.
    </div>
    """,
    unsafe_allow_html=True,
)
