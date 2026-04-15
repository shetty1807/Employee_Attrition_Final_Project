import io
import smtplib
from email.mime.text import MIMEText
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from fpdf import FPDF

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="Employee Attrition AI + HR Alert System",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------
# FORCE REMOVE FOOTER / HR LINE
# ---------------------------------
st.markdown("""
<style>
footer {visibility: hidden;}
hr {display: none;}
.small-note {display: none;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# CUSTOM CSS
# ---------------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1rem;
}
.metric-card {
    background: #111827;
    padding: 18px;
    border-radius: 14px;
    color: white;
    border: 1px solid #1f2937;
}
.metric-title {
    font-size: 15px;
    color: #cbd5e1;
}
.metric-value {
    font-size: 28px;
    font-weight: 700;
}
.section-card {
    background: #f8fafc;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
}
.notice-box {
    background: #fff7ed;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #fdba74;
}
.email-box {
    background: #eff6ff;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #93c5fd;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# PATHS
# ---------------------------------
CURRENT_FILE = Path(__file__).resolve()
BASE_DIR = CURRENT_FILE.parent.parent

DATA_PATH = BASE_DIR / "data" / "HR_Employee_Attrition.csv"
MODEL_PATH = BASE_DIR / "saved_models" / "xgb_attrition_model.pkl"
COLUMNS_PATH = BASE_DIR / "saved_models" / "model_columns.pkl"
METRICS_PATH = BASE_DIR / "saved_models" / "model_metrics.pkl"
CONF_MATRIX_NPY_PATH = BASE_DIR / "saved_models" / "conf_matrix.npy"
ACTION_TRACKER_PATH = BASE_DIR / "saved_models" / "hr_action_tracker.csv"
PREDICTION_HISTORY_PATH = BASE_DIR / "saved_models" / "prediction_history.csv"

ACTION_TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------
# LOAD FUNCTIONS
# ---------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_columns():
    return joblib.load(COLUMNS_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        return joblib.load(METRICS_PATH)
    return {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0
    }

@st.cache_data
def load_conf_matrix():
    if CONF_MATRIX_NPY_PATH.exists():
        return np.load(CONF_MATRIX_NPY_PATH)
    return np.array([[0, 0], [0, 0]])

# ---------------------------------
# HELPER FUNCTIONS
# ---------------------------------
def risk_label(probability: float) -> str:
    if probability < 0.35:
        return "Low Risk"
    elif probability < 0.65:
        return "Medium Risk"
    return "High Risk"

def priority_label(probability: float) -> str:
    if probability >= 0.80:
        return "Critical"
    elif probability >= 0.65:
        return "High"
    elif probability >= 0.35:
        return "Medium"
    return "Low"

def assign_hr_owner(risk_level: str) -> str:
    if risk_level == "High Risk":
        return "HR Manager"
    elif risk_level == "Medium Risk":
        return "Team Lead + HR Executive"
    return "Reporting Manager"

def escalation_flag(probability, overtime, job_satisfaction, work_life_balance):
    if probability >= 0.80 or (
        overtime == "Yes" and job_satisfaction <= 2 and work_life_balance <= 2
    ):
        return "Escalate Immediately"
    elif probability >= 0.65:
        return "Priority Review"
    elif probability >= 0.35:
        return "Monitor Closely"
    return "Normal Review"

def generate_ai_recommendations(
    probability,
    overtime,
    job_satisfaction,
    monthly_income,
    work_life_balance,
    training_times_last_year,
    years_since_last_promotion,
    environment_satisfaction,
    distance_from_home,
    num_companies_worked
):
    issues = []
    hr_actions = []
    manager_actions = []
    retention_plan = []
    risk_score = 0

    if overtime == "Yes":
        issues.append("Employee is working overtime frequently.")
        hr_actions.append("Review workload policy and discuss burnout risk with HR.")
        manager_actions.append("Reduce overtime burden and rebalance task allocation.")
        risk_score += 2

    if job_satisfaction <= 2:
        issues.append("Job satisfaction is low.")
        hr_actions.append("Schedule a retention discussion to understand dissatisfaction.")
        manager_actions.append("Conduct a one-to-one meeting and improve role clarity.")
        risk_score += 2

    if monthly_income < 4000:
        issues.append("Monthly income is comparatively low.")
        hr_actions.append("Review compensation, benefits, and retention incentives.")
        manager_actions.append("Support compensation justification with performance context.")
        risk_score += 1

    if work_life_balance <= 2:
        issues.append("Work-life balance is poor.")
        hr_actions.append("Consider flexible work arrangements or wellbeing support.")
        manager_actions.append("Adjust deadlines and reduce unnecessary pressure.")
        risk_score += 2

    if training_times_last_year <= 1:
        issues.append("Employee has received limited training.")
        hr_actions.append("Provide learning and development opportunities.")
        manager_actions.append("Nominate the employee for technical or role-based training.")
        risk_score += 1

    if years_since_last_promotion >= 5:
        issues.append("Employee has not been promoted for a long time.")
        hr_actions.append("Review promotion eligibility and career growth opportunities.")
        manager_actions.append("Discuss future growth path and internal mobility options.")
        risk_score += 2

    if environment_satisfaction <= 2:
        issues.append("Work environment satisfaction is low.")
        hr_actions.append("Assess team culture and workplace concerns.")
        manager_actions.append("Improve team engagement and manager support.")
        risk_score += 2

    if distance_from_home >= 20:
        issues.append("Employee travels a long distance to work.")
        hr_actions.append("Consider transport support or flexible work options.")
        manager_actions.append("Allow hybrid work where possible.")
        risk_score += 1

    if num_companies_worked >= 5:
        issues.append("Employee has changed jobs frequently in the past.")
        hr_actions.append("Strengthen retention engagement and long-term career planning.")
        manager_actions.append("Build stronger connection through frequent check-ins.")
        risk_score += 1

    if not issues:
        issues.append("No major retention issues detected from the current profile.")
        hr_actions.append("Continue regular employee engagement and recognition.")
        manager_actions.append("Maintain supportive management and periodic check-ins.")

    hr_actions = list(dict.fromkeys(hr_actions))
    manager_actions = list(dict.fromkeys(manager_actions))

    if probability >= 0.80 or risk_score >= 8:
        priority = "Critical"
        confidence = "High"
        retention_plan = [
            "Within 48 hours: HR retention meeting",
            "Within 3 days: Manager one-to-one discussion",
            "Within 1 week: Workload / salary / promotion review",
            "Within 2 weeks: Track improvement and follow-up action"
        ]
    elif probability >= 0.65 or risk_score >= 5:
        priority = "High"
        confidence = "High"
        retention_plan = [
            "Within 1 week: HR discussion",
            "Within 1 week: Manager workload review",
            "Within 2 weeks: Career growth or training plan",
            "Within 1 month: Monitor retention signals"
        ]
    elif probability >= 0.35 or risk_score >= 3:
        priority = "Medium"
        confidence = "Medium"
        retention_plan = [
            "Within 2 weeks: Manager check-in",
            "Within 2 weeks: Engagement review",
            "Within 1 month: Training or wellbeing support",
            "Monitor monthly for changes"
        ]
    else:
        priority = "Low"
        confidence = "Medium"
        retention_plan = [
            "Continue regular support",
            "Maintain employee recognition",
            "Review satisfaction periodically"
        ]

    main_reason = issues[0] if len(issues) > 0 else "No major issue identified."
    secondary_reason = issues[1] if len(issues) > 1 else "No secondary issue identified."

    return {
        "priority": priority,
        "confidence": confidence,
        "main_reason": main_reason,
        "secondary_reason": secondary_reason,
        "issues": issues,
        "hr_actions": hr_actions,
        "manager_actions": manager_actions,
        "retention_plan": retention_plan
    }

def generate_risk_reasons(
    overtime,
    job_satisfaction,
    monthly_income,
    work_life_balance,
    years_since_last_promotion,
    environment_satisfaction,
    distance_from_home,
    num_companies_worked
):
    reasons = []

    if overtime == "Yes":
        reasons.append(("OverTime", "Frequent overtime may increase burnout risk"))
    if job_satisfaction <= 2:
        reasons.append(("JobSatisfaction", "Low job satisfaction may lead to resignation"))
    if monthly_income < 4000:
        reasons.append(("MonthlyIncome", "Lower salary may reduce retention"))
    if work_life_balance <= 2:
        reasons.append(("WorkLifeBalance", "Poor work-life balance may push attrition"))
    if years_since_last_promotion >= 5:
        reasons.append(("YearsSinceLastPromotion", "Long promotion gap may affect motivation"))
    if environment_satisfaction <= 2:
        reasons.append(("EnvironmentSatisfaction", "Low environment satisfaction is a risk factor"))
    if distance_from_home >= 20:
        reasons.append(("DistanceFromHome", "Long commuting distance may affect retention"))
    if num_companies_worked >= 5:
        reasons.append(("NumCompaniesWorked", "Frequent job changes suggest higher switch tendency"))

    if not reasons:
        reasons.append(("General", "Current profile shows relatively stable retention indicators"))

    return reasons[:5]

def generate_hr_notice(employee_name, probability, risk_level, issues, recommendations):
    notice = f"""
HR ALERT NOTICE
----------------------------------------
Employee Name: {employee_name}
Attrition Probability: {probability:.2%}
Risk Level: {risk_level}

Detected Issues:
"""
    for issue in issues:
        notice += f"- {issue}\n"

    notice += "\nRecommended HR Actions:\n"
    for rec in recommendations:
        notice += f"- {rec}\n"

    if risk_level == "High Risk":
        notice += "\nAction Required: Immediate HR intervention is recommended."
    elif risk_level == "Medium Risk":
        notice += "\nAction Required: Monitor employee closely and schedule manager review."
    else:
        notice += "\nAction Required: Continue regular engagement and support."

    return notice

def create_single_prediction_report(
    input_summary_df,
    employee_name,
    probability,
    label,
    ai_result,
    reasons,
    hr_owner,
    escalation_status
):
    output = io.StringIO()
    output.write("EMPLOYEE ATTRITION RISK REPORT\n")
    output.write("=" * 45 + "\n\n")
    output.write(f"Employee Name: {employee_name}\n")
    output.write(f"Predicted Attrition Probability: {probability:.4f}\n")
    output.write(f"Risk Level: {label}\n")
    output.write(f"Priority Level: {ai_result['priority']}\n")
    output.write(f"Recommendation Confidence: {ai_result['confidence']}\n")
    output.write(f"Main Reason: {ai_result['main_reason']}\n")
    output.write(f"Secondary Reason: {ai_result['secondary_reason']}\n")
    output.write(f"Assigned HR Owner: {hr_owner}\n")
    output.write(f"Escalation Status: {escalation_status}\n\n")

    output.write("TOP RISK REASONS\n")
    output.write("-" * 22 + "\n")
    for _, reason_text in reasons:
        output.write(f"- {reason_text}\n")

    output.write("\nDETECTED ISSUES\n")
    output.write("-" * 17 + "\n")
    for issue in ai_result["issues"]:
        output.write(f"- {issue}\n")

    output.write("\nHR ACTIONS\n")
    output.write("-" * 17 + "\n")
    for rec in ai_result["hr_actions"]:
        output.write(f"- {rec}\n")

    output.write("\nMANAGER ACTIONS\n")
    output.write("-" * 17 + "\n")
    for rec in ai_result["manager_actions"]:
        output.write(f"- {rec}\n")

    output.write("\nRETENTION PLAN\n")
    output.write("-" * 17 + "\n")
    for step in ai_result["retention_plan"]:
        output.write(f"- {step}\n")

    output.write("\nEMPLOYEE INPUT SUMMARY\n")
    output.write("-" * 24 + "\n")
    output.write(input_summary_df.to_string(index=False))

    return output.getvalue()

def create_pdf_report(employee_name, probability, risk_level, priority, hr_owner, escalation_status, reasons, recommendations):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    def clean_text(text):
        if text is None:
            return ""
        text = str(text)
        text = text.replace("–", "-").replace("—", "-").replace("’", "'").replace("“", '"').replace("”", '"')
        return text.encode("latin-1", "replace").decode("latin-1")

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, clean_text("Employee Attrition Risk Report"), ln=True, align="C")
    pdf.ln(4)

    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, clean_text(f"Employee Name: {employee_name}"), ln=True)
    pdf.cell(0, 8, clean_text(f"Attrition Probability: {probability:.2%}"), ln=True)
    pdf.cell(0, 8, clean_text(f"Risk Level: {risk_level}"), ln=True)
    pdf.cell(0, 8, clean_text(f"Priority: {priority}"), ln=True)
    pdf.cell(0, 8, clean_text(f"HR Owner: {hr_owner}"), ln=True)
    pdf.cell(0, 8, clean_text(f"Escalation Status: {escalation_status}"), ln=True)
    pdf.ln(4)

    usable_width = 180

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, clean_text("Top Risk Reasons"), ln=True)
    pdf.set_font("Arial", size=11)
    for _, reason_text in reasons:
        pdf.multi_cell(usable_width, 8, clean_text(f"- {reason_text}"))
        pdf.ln(1)

    pdf.ln(2)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, clean_text("Recommended Actions"), ln=True)
    pdf.set_font("Arial", size=11)
    for rec in recommendations:
        pdf.multi_cell(usable_width, 8, clean_text(f"- {rec}"))
        pdf.ln(1)

    pdf_output = pdf.output(dest="S")
    if isinstance(pdf_output, str):
        return pdf_output.encode("latin-1", "replace")
    return bytes(pdf_output)

def send_email_alert(sender_email, sender_password, receiver_email, subject, body):
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = receiver_email

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, [receiver_email], msg.as_string())
        server.quit()
        return True, "Email sent successfully."
    except Exception as e:
        return False, f"Email failed: {e}"

def add_metric_card(column, title, value):
    with column:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">{title}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

def plot_bar_chart(series, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8, 4))
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig)

def preprocess_bulk_data(bulk_df, model_columns):
    cols_to_drop = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
    bulk_df = bulk_df.drop(columns=cols_to_drop, errors="ignore")
    bulk_df = bulk_df.drop(columns=["Attrition"], errors="ignore")
    bulk_df = pd.get_dummies(bulk_df, drop_first=True)

    for col in model_columns:
        if col not in bulk_df.columns:
            bulk_df[col] = 0

    bulk_df = bulk_df[model_columns]
    return bulk_df

def load_action_tracker():
    if ACTION_TRACKER_PATH.exists():
        return pd.read_csv(ACTION_TRACKER_PATH)
    return pd.DataFrame(columns=[
        "Employee Name", "Probability", "Risk Level",
        "Priority", "HR Owner", "Status"
    ])

def save_action_tracker(tracker_df):
    ACTION_TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    tracker_df.to_csv(ACTION_TRACKER_PATH, index=False)

def load_prediction_history():
    if PREDICTION_HISTORY_PATH.exists():
        return pd.read_csv(PREDICTION_HISTORY_PATH)
    return pd.DataFrame(columns=[
        "Employee Name",
        "Probability",
        "Risk Level",
        "Priority",
        "HR Owner",
        "Timestamp"
    ])

def save_prediction_history(history_df):
    PREDICTION_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(PREDICTION_HISTORY_PATH, index=False)

def chatbot_response(user_query):
    query = user_query.lower().strip()

    if "attrition" in query:
        return "Employee attrition means employees leaving the organization. This system predicts attrition risk and helps HR take preventive action."
    elif "high risk" in query:
        return "High Risk means the employee has a high probability of leaving the company. HR should review workload, job satisfaction, salary, promotion gap, and work-life balance."
    elif "medium risk" in query:
        return "Medium Risk means the employee may be at moderate risk of leaving. HR should monitor the employee and take supportive actions early."
    elif "low risk" in query:
        return "Low Risk means the employee currently shows stable retention signs. Regular support and engagement should continue."
    elif "hr" in query or "what should hr do" in query:
        return "HR should review employee satisfaction, overtime, salary, promotion opportunities, training, and manager support. The recommendation engine in this project helps with that."
    elif "overtime" in query:
        return "Frequent overtime may increase stress and burnout, which can raise attrition risk."
    elif "salary" in query or "income" in query:
        return "Low monthly income compared to workload or market expectation can reduce retention and increase attrition risk."
    elif "promotion" in query:
        return "A long gap since last promotion may reduce employee motivation and increase the chance of attrition."
    elif "work life balance" in query or "work-life balance" in query:
        return "Poor work-life balance is a major reason employees may leave. Flexible policies and workload control can help."
    elif "bulk prediction" in query:
        return "Bulk Prediction allows HR to upload a CSV file and predict attrition risk for many employees at once."
    elif "action tracker" in query:
        return "HR Action Tracker helps record follow-up actions, assign ownership, and update employee intervention status."
    elif "shap" in query or "explainability" in query:
        return "SHAP Explainability shows which features influenced the model prediction the most for a selected employee."
    elif "dashboard" in query:
        return "Dashboard Analytics shows attrition trends, department-wise patterns, overtime impact, and other useful visual insights."
    elif "hello" in query or "hi" in query:
        return "Hello! I am your HR Attrition Assistant. Ask me about attrition, risk levels, HR actions, bulk prediction, SHAP, or dashboard analytics."
    else:
        return "I can help with attrition, risk levels, HR actions, overtime, salary, promotions, dashboard analytics, SHAP explainability, bulk prediction, and action tracker."

def build_dashboard_filtered_data(df):
    filtered_df = df.copy()

    department_options = ["All"] + sorted(df["Department"].dropna().astype(str).unique().tolist()) if "Department" in df.columns else ["All"]
    attrition_options = ["All"] + sorted(df["Attrition"].dropna().astype(str).unique().tolist()) if "Attrition" in df.columns else ["All"]

    selected_department = st.selectbox("Filter by Department", department_options)
    selected_attrition = st.selectbox("Filter by Attrition", attrition_options)

    if selected_department != "All" and "Department" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Department"].astype(str) == selected_department]

    if selected_attrition != "All" and "Attrition" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Attrition"].astype(str) == selected_attrition]

    return filtered_df

def build_dashboard_risky_profiles(filtered_dashboard_df):
    if "OverTime" in filtered_dashboard_df.columns and "JobSatisfaction" in filtered_dashboard_df.columns:
        return filtered_dashboard_df[
            (filtered_dashboard_df["OverTime"].astype(str) == "Yes") &
            (filtered_dashboard_df["JobSatisfaction"] <= 2)
        ].copy()
    return pd.DataFrame()

def add_dashboard_risky_to_tracker(risky_profiles_df):
    if risky_profiles_df.empty:
        return 0

    tracker_df = load_action_tracker()
    added_count = 0

    for idx, row in risky_profiles_df.iterrows():
        employee_name = row.get("EmployeeName", f"Employee_{idx+1}")
        department = str(row.get("Department", "Unknown"))
        risk_level = "High Risk"
        probability = 0.80
        hr_owner = assign_hr_owner(risk_level)

        new_row = pd.DataFrame([{
            "Employee Name": employee_name,
            "Probability": probability,
            "Risk Level": risk_level,
            "Priority": "High",
            "HR Owner": hr_owner,
            "Status": "Pending"
        }])

        tracker_df = pd.concat([tracker_df, new_row], ignore_index=True)
        added_count += 1

    tracker_df = tracker_df.drop_duplicates(subset=["Employee Name"], keep="last")
    save_action_tracker(tracker_df)
    return added_count

# ---------------------------------
# FILE CHECK
# ---------------------------------
missing_files = []

if not DATA_PATH.exists():
    missing_files.append(f"Dataset file missing: {DATA_PATH}")
if not MODEL_PATH.exists():
    missing_files.append(f"Model file missing: {MODEL_PATH}")
if not COLUMNS_PATH.exists():
    missing_files.append(f"Columns file missing: {COLUMNS_PATH}")

if missing_files:
    st.error("Some required files are missing.")
    for item in missing_files:
        st.write("-", item)

    st.info("Keep these files in your project like this:")
    st.code(
        """Employee_Attrition_Final_Project/
├── app/
│   └── app.py
├── data/
│   └── HR_Employee_Attrition.csv
└── saved_models/
    ├── xgb_attrition_model.pkl
    ├── model_columns.pkl
    ├── model_metrics.pkl
    └── conf_matrix.npy""",
        language="text"
    )
    st.stop()

# ---------------------------------
# LOAD FILES
# ---------------------------------
model = load_model()
model_columns = load_columns()
df = load_data()
metrics = load_metrics()
conf_matrix = load_conf_matrix()

# ---------------------------------
# ROLE-BASED LOGIN
# ---------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_role" not in st.session_state:
    st.session_state.user_role = ""

if "display_name" not in st.session_state:
    st.session_state.display_name = ""

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

USERS = {
    "Admin": {
        "username": "admin",
        "password": "1234",
        "display_name": "System Admin"
    },
    "HR": {
        "username": "hr",
        "password": "hr123",
        "display_name": "HR Manager"
    },
    "Manager": {
        "username": "manager",
        "password": "mgr123",
        "display_name": "Reporting Manager"
    }
}

if not st.session_state.logged_in:
    st.markdown("""
        <div style='padding:20px; background-color:#0f172a; border-radius:12px; margin-bottom:20px;'>
            <h1 style='color:white; margin-bottom:8px;'>Login Page</h1>
            <p style='color:#cbd5e1; margin-top:0;'>AI-Powered Employee Attrition & HR Alert System</p>
        </div>
    """, unsafe_allow_html=True)

    role = st.selectbox("Select Role", ["Admin", "HR", "Manager"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns([1, 5])
    with col1:
        login_clicked = st.button("Login")

    if login_clicked:
        selected_user = USERS[role]
        if username == selected_user["username"] and password == selected_user["password"]:
            st.session_state.logged_in = True
            st.session_state.user_role = role
            st.session_state.display_name = selected_user["display_name"]
            st.success(f"Login successful as {role}")
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.info("Demo credentials: Admin/admin/1234, HR/hr/hr123, Manager/manager/mgr123")
    st.stop()

# ---------------------------------
# BASIC DATA PREP
# ---------------------------------
df_clean = df.copy()

if "Attrition" in df_clean.columns:
    attrition_series = df_clean["Attrition"].astype(str)
else:
    attrition_series = pd.Series([], dtype=str)

attrition_rate = 0.0
if not attrition_series.empty:
    yes_mask = attrition_series.str.lower().eq("yes")
    attrition_rate = yes_mask.mean() * 100

avg_income = df_clean["MonthlyIncome"].mean() if "MonthlyIncome" in df_clean.columns else 0
avg_years = df_clean["YearsAtCompany"].mean() if "YearsAtCompany" in df_clean.columns else 0

# ---------------------------------
# HEADER
# ---------------------------------
st.markdown(
    """
    <div style='padding:18px; background-color:#0f172a; border-radius:12px;'>
        <h1 style='color:white; margin-bottom:0;'>AI-Powered Employee Attrition & HR Alert System</h1>
        <p style='color:#cbd5e1; margin-top:8px;'>Predict risk, recommend retention action, and notify HR for early intervention</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.success(f"Welcome, {st.session_state.display_name} ({st.session_state.user_role})")
st.write("")

# ---------------------------------
# SIDEBAR
# ---------------------------------
if st.session_state.user_role == "Admin":
    menu_options = [
        "Project Overview",
        "Dashboard Analytics",
        "Dataset Insights",
        "Model Comparison",
        "Model Performance",
        "Prediction",
        "Prediction History",
        "Bulk Prediction",
        "HR Alert Center",
        "HR Action Tracker",
        "AI Recommendations",
        "HR Chatbot",
        "Feature Importance",
        "SHAP Explainability"
    ]
elif st.session_state.user_role == "HR":
    menu_options = [
        "Dashboard Analytics",
        "Prediction",
        "Prediction History",
        "Bulk Prediction",
        "HR Alert Center",
        "HR Action Tracker",
        "AI Recommendations",
        "HR Chatbot"
    ]
elif st.session_state.user_role == "Manager":
    menu_options = [
        "Dashboard Analytics",
        "Prediction",
        "Prediction History",
        "HR Chatbot"
    ]
else:
    menu_options = ["Dashboard Analytics"]

menu = st.sidebar.radio("Navigation", menu_options)

st.sidebar.markdown("---")
st.sidebar.write(f"Logged in as: **{st.session_state.display_name}**")
st.sidebar.write(f"Role: **{st.session_state.user_role}**")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.user_role = ""
    st.session_state.display_name = ""
    st.session_state.last_prediction = None
    st.session_state.chat_history = []
    st.rerun()

# ---------------------------------
# PAGE 1: PROJECT OVERVIEW
# ---------------------------------
if menu == "Project Overview":
    st.header("Project Overview")

    c1, c2, c3, c4 = st.columns(4)
    add_metric_card(c1, "Total Employees", f"{df.shape[0]}")
    add_metric_card(c2, "Total Features", f"{df.shape[1]}")
    add_metric_card(c3, "Selected Model", "XGBoost")
    add_metric_card(c4, "Attrition Rate", f"{attrition_rate:.2f}%")

    st.write("")
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Problem Statement")
        st.write(
            """
            Employee attrition leads to higher hiring cost, loss of skilled workers,
            and lower productivity. This project predicts attrition risk and helps
            HR teams take early preventive action.
            """
        )

        st.subheader("Objective")
        st.write(
            """
            To predict employee attrition using machine learning and support HR
            through risk prioritization, retention recommendations, alert notices,
            email alerts, action tracking, prediction history, chatbot support,
            and dashboard-based risky employee workflows.
            """
        )

        st.subheader("System Workflow")
        st.write(
            """
            Dataset → Data Cleaning → Encoding → Model Training →
            Prediction → Risk Prioritization → HR Recommendation →
            HR Alert Notice Generation → Email / Action Tracking / History
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        if not attrition_series.empty:
            fig, ax = plt.subplots(figsize=(5, 4))
            attrition_series.value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")
            ax.set_title("Attrition Distribution")
            st.pyplot(fig)

# ---------------------------------
# PAGE 2: DASHBOARD ANALYTICS
# ---------------------------------
elif menu == "Dashboard Analytics":
    st.header("Dashboard Analytics")

    filtered_dashboard_df = build_dashboard_filtered_data(df)

    high_risk_proxy = 0
    if "OverTime" in filtered_dashboard_df.columns and "JobSatisfaction" in filtered_dashboard_df.columns:
        high_risk_proxy = filtered_dashboard_df[
            (filtered_dashboard_df["OverTime"].astype(str) == "Yes") &
            (filtered_dashboard_df["JobSatisfaction"] <= 2)
        ].shape[0]

    c1, c2, c3, c4 = st.columns(4)
    add_metric_card(c1, "Attrition Rate", f"{attrition_rate:.2f}%")
    add_metric_card(c2, "Average Monthly Income", f"{avg_income:.0f}")
    add_metric_card(c3, "Average Years at Company", f"{avg_years:.1f}")
    add_metric_card(c4, "Potential High-Risk Profiles", f"{high_risk_proxy}")

    st.write("")
    col1, col2 = st.columns(2)

    with col1:
        if "Department" in filtered_dashboard_df.columns and "Attrition" in filtered_dashboard_df.columns:
            dept_attrition = pd.crosstab(filtered_dashboard_df["Department"], filtered_dashboard_df["Attrition"])
            fig, ax = plt.subplots(figsize=(7, 4))
            dept_attrition.plot(kind="bar", ax=ax)
            ax.set_title("Department-wise Attrition")
            ax.set_xlabel("Department")
            ax.set_ylabel("Count")
            plt.xticks(rotation=25, ha="right")
            st.pyplot(fig)

    with col2:
        if "OverTime" in filtered_dashboard_df.columns and "Attrition" in filtered_dashboard_df.columns:
            overtime_attrition = pd.crosstab(filtered_dashboard_df["OverTime"], filtered_dashboard_df["Attrition"])
            fig, ax = plt.subplots(figsize=(7, 4))
            overtime_attrition.plot(kind="bar", ax=ax)
            ax.set_title("OverTime vs Attrition")
            ax.set_xlabel("OverTime")
            ax.set_ylabel("Count")
            plt.xticks(rotation=0)
            st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        if "JobRole" in filtered_dashboard_df.columns:
            plot_bar_chart(
                filtered_dashboard_df["JobRole"].value_counts().head(10),
                "Top Job Roles Distribution",
                "Job Role",
                "Count"
            )

    with col4:
        if "WorkLifeBalance" in filtered_dashboard_df.columns and "Attrition" in filtered_dashboard_df.columns:
            wlb_attrition = pd.crosstab(filtered_dashboard_df["WorkLifeBalance"], filtered_dashboard_df["Attrition"])
            fig, ax = plt.subplots(figsize=(7, 4))
            wlb_attrition.plot(kind="bar", ax=ax)
            ax.set_title("Work-Life Balance vs Attrition")
            ax.set_xlabel("Work-Life Balance")
            ax.set_ylabel("Count")
            plt.xticks(rotation=0)
            st.pyplot(fig)

    risky_profiles = build_dashboard_risky_profiles(filtered_dashboard_df)

    st.subheader("Auto High-Risk Alert Section")

    if risky_profiles.empty:
        st.info("No risky employees found for the current dashboard filters.")
    else:
        st.warning(f"Found {len(risky_profiles)} potentially risky employees based on overtime + low job satisfaction.")

        st.subheader("Potential Risky Employees")
        st.dataframe(risky_profiles.head(10), use_container_width=True)

        st.download_button(
            label="Download Risky Employees Report",
            data=risky_profiles.to_csv(index=False).encode("utf-8"),
            file_name="dashboard_risky_employees.csv",
            mime="text/csv"
        )

        if st.button("Add Risky Employees to HR Action Tracker"):
            added_count = add_dashboard_risky_to_tracker(risky_profiles)
            st.success(f"{added_count} risky employees were added to HR Action Tracker.")

    st.subheader("Filtered Employee Data")
    st.dataframe(filtered_dashboard_df, use_container_width=True)

    st.download_button(
        label="Download Filtered Dashboard Data",
        data=filtered_dashboard_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_dashboard_data.csv",
        mime="text/csv"
    )

# ---------------------------------
# PAGE 3: DATASET INSIGHTS
# ---------------------------------
elif menu == "Dataset Insights":
    st.header("Dataset Insights")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    info1, info2 = st.columns(2)
    with info1:
        st.subheader("Dataset Shape")
        st.write(df.shape)

    with info2:
        st.subheader("Missing Values Summary")
        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ["Column", "Missing Values"]
        st.dataframe(missing_df, use_container_width=True)

    ch1, ch2 = st.columns(2)

    with ch1:
        if "Age" in df.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(df["Age"], bins=15)
            ax.set_title("Age Distribution")
            ax.set_xlabel("Age")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

    with ch2:
        if "MonthlyIncome" in df.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(df["MonthlyIncome"], bins=15)
            ax.set_title("Monthly Income Distribution")
            ax.set_xlabel("Monthly Income")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

    ch3, ch4 = st.columns(2)

    with ch3:
        if "Attrition" in df.columns:
            attrition_counts = df["Attrition"].value_counts()
            fig, ax = plt.subplots(figsize=(7, 4))
            attrition_counts.plot(kind="bar", ax=ax)
            ax.set_title("Employee Attrition Count")
            ax.set_xlabel("Attrition")
            ax.set_ylabel("Count")
            plt.xticks(rotation=0)
            st.pyplot(fig)

    with ch4:
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.shape[1] > 1:
            corr = numeric_df.corr().round(2)
            fig, ax = plt.subplots(figsize=(8, 5))
            im = ax.imshow(corr, aspect="auto")
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90)
            ax.set_yticklabels(corr.columns)
            ax.set_title("Correlation Heatmap")
            plt.colorbar(im, ax=ax)
            st.pyplot(fig)

# ---------------------------------
# PAGE 4: MODEL COMPARISON
# ---------------------------------
elif menu == "Model Comparison":
    st.header("Model Comparison")

    comparison_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "SVM", "XGBoost"],
        "Accuracy": [0.84, 0.87, 0.85, float(metrics.get("accuracy", 0.88))],
        "Precision": [0.79, 0.84, 0.81, float(metrics.get("precision", 0.86))],
        "Recall": [0.74, 0.78, 0.76, float(metrics.get("recall", 0.82))],
        "F1 Score": [0.76, 0.81, 0.78, float(metrics.get("f1_score", 0.84))]
    })

    st.info("Replace the sample values for Logistic Regression, Random Forest, and SVM with your real model results if you train them.")
    st.dataframe(comparison_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(comparison_df["Model"], comparison_df["Accuracy"])
    ax.set_title("Model Accuracy Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    plt.xticks(rotation=20, ha="right")
    st.pyplot(fig)

    best_model = comparison_df.sort_values("Accuracy", ascending=False).iloc[0]["Model"]
    st.success(f"Best performing model selected for deployment: {best_model}")

# ---------------------------------
# PAGE 5: MODEL PERFORMANCE
# ---------------------------------
elif menu == "Model Performance":
    st.header("Model Performance")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics.get('accuracy', 0):.2f}")
    c2.metric("Precision", f"{metrics.get('precision', 0):.2f}")
    c3.metric("Recall", f"{metrics.get('recall', 0):.2f}")
    c4.metric("F1 Score", f"{metrics.get('f1_score', 0):.2f}")

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(conf_matrix, cmap="Blues")
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, conf_matrix[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("XGBoost Confusion Matrix")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No", "Yes"])
    ax.set_yticklabels(["No", "Yes"])
    plt.colorbar(im, ax=ax)
    st.pyplot(fig)

# ---------------------------------
# PAGE 6: SINGLE PREDICTION
# ---------------------------------
elif menu == "Prediction":
    st.header("Employee Attrition Prediction")
    st.write("Enter employee details below to predict attrition risk and generate HR notice.")

    employee_name = st.text_input("Employee Name", value="Employee 001")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 60, 30)
        daily_rate = st.number_input("Daily Rate", 100, 2000, 800)
        distance_from_home = st.slider("Distance From Home", 1, 30, 5)
        education = st.slider("Education", 1, 5, 3)
        environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 2)
        job_involvement = st.slider("Job Involvement", 1, 4, 2)
        job_level = st.slider("Job Level", 1, 5, 2)
        job_satisfaction = st.slider("Job Satisfaction", 1, 4, 2)

    with col2:
        monthly_income = st.number_input("Monthly Income", 1000, 50000, 5000)
        monthly_rate = st.number_input("Monthly Rate", 1000, 30000, 10000)
        num_companies_worked = st.slider("Num Companies Worked", 0, 10, 2)
        percent_salary_hike = st.slider("Percent Salary Hike", 10, 30, 15)
        performance_rating = st.slider("Performance Rating", 1, 4, 3)
        relationship_satisfaction = st.slider("Relationship Satisfaction", 1, 4, 2)
        stock_option_level = st.slider("Stock Option Level", 0, 3, 1)
        total_working_years = st.slider("Total Working Years", 0, 40, 10)

    with col3:
        training_times_last_year = st.slider("Training Times Last Year", 0, 10, 2)
        work_life_balance = st.slider("Work Life Balance", 1, 4, 2)
        years_at_company = st.slider("Years At Company", 0, 40, 5)
        years_in_current_role = st.slider("Years In Current Role", 0, 20, 3)
        years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)
        years_with_curr_manager = st.slider("Years With Current Manager", 0, 20, 3)

    st.subheader("Categorical Details")

    col4, col5, col6 = st.columns(3)

    with col4:
        business_travel = st.selectbox(
            "Business Travel",
            ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
        )
        department = st.selectbox(
            "Department",
            ["Sales", "Research & Development", "Human Resources"]
        )

    with col5:
        education_field = st.selectbox(
            "Education Field",
            ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"]
        )
        gender = st.selectbox("Gender", ["Male", "Female"])

    with col6:
        job_role = st.selectbox(
            "Job Role",
            [
                "Sales Executive",
                "Research Scientist",
                "Laboratory Technician",
                "Manufacturing Director",
                "Healthcare Representative",
                "Manager",
                "Sales Representative",
                "Research Director",
                "Human Resources"
            ]
        )
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        overtime = st.selectbox("OverTime", ["Yes", "No"])

    if st.button("Predict Attrition"):
        input_dict = {
            "Age": age,
            "DailyRate": daily_rate,
            "DistanceFromHome": distance_from_home,
            "Education": education,
            "EnvironmentSatisfaction": environment_satisfaction,
            "JobInvolvement": job_involvement,
            "JobLevel": job_level,
            "JobSatisfaction": job_satisfaction,
            "MonthlyIncome": monthly_income,
            "MonthlyRate": monthly_rate,
            "NumCompaniesWorked": num_companies_worked,
            "PercentSalaryHike": percent_salary_hike,
            "PerformanceRating": performance_rating,
            "RelationshipSatisfaction": relationship_satisfaction,
            "StockOptionLevel": stock_option_level,
            "TotalWorkingYears": total_working_years,
            "TrainingTimesLastYear": training_times_last_year,
            "WorkLifeBalance": work_life_balance,
            "YearsAtCompany": years_at_company,
            "YearsInCurrentRole": years_in_current_role,
            "YearsSinceLastPromotion": years_since_last_promotion,
            "YearsWithCurrManager": years_with_curr_manager,

            "BusinessTravel_Travel_Frequently": 1 if business_travel == "Travel_Frequently" else 0,
            "BusinessTravel_Travel_Rarely": 1 if business_travel == "Travel_Rarely" else 0,

            "Department_Research & Development": 1 if department == "Research & Development" else 0,
            "Department_Sales": 1 if department == "Sales" else 0,

            "EducationField_Life Sciences": 1 if education_field == "Life Sciences" else 0,
            "EducationField_Marketing": 1 if education_field == "Marketing" else 0,
            "EducationField_Medical": 1 if education_field == "Medical" else 0,
            "EducationField_Other": 1 if education_field == "Other" else 0,
            "EducationField_Technical Degree": 1 if education_field == "Technical Degree" else 0,

            "Gender_Male": 1 if gender == "Male" else 0,

            "JobRole_Human Resources": 1 if job_role == "Human Resources" else 0,
            "JobRole_Laboratory Technician": 1 if job_role == "Laboratory Technician" else 0,
            "JobRole_Manager": 1 if job_role == "Manager" else 0,
            "JobRole_Manufacturing Director": 1 if job_role == "Manufacturing Director" else 0,
            "JobRole_Research Director": 1 if job_role == "Research Director" else 0,
            "JobRole_Research Scientist": 1 if job_role == "Research Scientist" else 0,
            "JobRole_Sales Executive": 1 if job_role == "Sales Executive" else 0,
            "JobRole_Sales Representative": 1 if job_role == "Sales Representative" else 0,

            "MaritalStatus_Married": 1 if marital_status == "Married" else 0,
            "MaritalStatus_Single": 1 if marital_status == "Single" else 0,

            "OverTime_Yes": 1 if overtime == "Yes" else 0
        }

        input_df = pd.DataFrame([input_dict])

        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[model_columns]

        probability = model.predict_proba(input_df)[0][1]
        label = risk_label(probability)

        reasons = generate_risk_reasons(
            overtime=overtime,
            job_satisfaction=job_satisfaction,
            monthly_income=monthly_income,
            work_life_balance=work_life_balance,
            years_since_last_promotion=years_since_last_promotion,
            environment_satisfaction=environment_satisfaction,
            distance_from_home=distance_from_home,
            num_companies_worked=num_companies_worked
        )

        ai_result = generate_ai_recommendations(
            probability=probability,
            overtime=overtime,
            job_satisfaction=job_satisfaction,
            monthly_income=monthly_income,
            work_life_balance=work_life_balance,
            training_times_last_year=training_times_last_year,
            years_since_last_promotion=years_since_last_promotion,
            environment_satisfaction=environment_satisfaction,
            distance_from_home=distance_from_home,
            num_companies_worked=num_companies_worked
        )

        hr_owner = assign_hr_owner(label)
        escalation_status = escalation_flag(probability, overtime, job_satisfaction, work_life_balance)

        all_recommendations = ai_result["hr_actions"] + ai_result["manager_actions"]

        hr_notice = generate_hr_notice(
            employee_name=employee_name,
            probability=probability,
            risk_level=label,
            issues=ai_result["issues"],
            recommendations=all_recommendations
        )

        st.session_state.last_prediction = {
            "employee_name": employee_name,
            "probability": probability,
            "risk_level": label,
            "priority": ai_result["priority"],
            "hr_owner": hr_owner,
            "hr_notice": hr_notice
        }

        st.subheader("Prediction Result")

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Attrition Probability", f"{probability:.2%}")
        mc2.metric("Risk Level", label)
        mc3.metric("Priority", ai_result["priority"])
        mc4.metric("HR Owner", hr_owner)

        if label == "Low Risk":
            st.success("This employee currently shows low attrition risk.")
        elif label == "Medium Risk":
            st.warning("This employee shows medium attrition risk.")
        else:
            st.error("This employee shows high attrition risk.")

        st.progress(float(probability))
        st.info(f"Escalation Status: {escalation_status}")

        st.subheader("Risk Summary")
        st.write(f"**Priority:** {ai_result['priority']}")
        st.write(f"**Recommendation Confidence:** {ai_result['confidence']}")
        st.write(f"**Main Reason:** {ai_result['main_reason']}")
        st.write(f"**Secondary Reason:** {ai_result['secondary_reason']}")

        st.subheader("Top Reasons for Risk")
        reasons_df = pd.DataFrame(reasons, columns=["Factor", "Why it matters"])
        st.dataframe(reasons_df, use_container_width=True)

        st.subheader("Detected Issues")
        for issue in ai_result["issues"]:
            st.write(f"- {issue}")

        st.subheader("HR Actions")
        for rec in ai_result["hr_actions"]:
            st.write(f"- {rec}")

        st.subheader("Manager Actions")
        for rec in ai_result["manager_actions"]:
            st.write(f"- {rec}")

        st.subheader("Retention Plan")
        for step in ai_result["retention_plan"]:
            st.write(f"- {step}")

        st.subheader("Generated HR Notice")
        st.markdown('<div class="notice-box">', unsafe_allow_html=True)
        st.text_area("HR Notice Message", hr_notice, height=260)
        st.markdown("</div>", unsafe_allow_html=True)

        input_summary_df = pd.DataFrame({
            "Feature": [
                "Age", "DailyRate", "DistanceFromHome", "Education", "EnvironmentSatisfaction",
                "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate",
                "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
                "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
                "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
                "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
                "BusinessTravel", "Department", "EducationField", "Gender",
                "JobRole", "MaritalStatus", "OverTime"
            ],
            "Value": [
                age, daily_rate, distance_from_home, education, environment_satisfaction,
                job_involvement, job_level, job_satisfaction, monthly_income, monthly_rate,
                num_companies_worked, percent_salary_hike, performance_rating,
                relationship_satisfaction, stock_option_level, total_working_years,
                training_times_last_year, work_life_balance, years_at_company,
                years_in_current_role, years_since_last_promotion, years_with_curr_manager,
                business_travel, department, education_field, gender,
                job_role, marital_status, overtime
            ]
        })

        report_text = create_single_prediction_report(
            input_summary_df=input_summary_df,
            employee_name=employee_name,
            probability=probability,
            label=label,
            ai_result=ai_result,
            reasons=reasons,
            hr_owner=hr_owner,
            escalation_status=escalation_status
        )

        pdf_bytes = create_pdf_report(
            employee_name=employee_name,
            probability=probability,
            risk_level=label,
            priority=ai_result["priority"],
            hr_owner=hr_owner,
            escalation_status=escalation_status,
            reasons=reasons,
            recommendations=all_recommendations
        )

        st.download_button(
            label="Download Employee Risk Report",
            data=report_text,
            file_name="employee_attrition_risk_report.txt",
            mime="text/plain"
        )
        st.download_button(
            label="Download HR Notice",
            data=hr_notice,
            file_name="hr_notice.txt",
            mime="text/plain"
        )
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name="employee_attrition_report.pdf",
            mime="application/pdf"
        )

        tracker_df = load_action_tracker()
        new_row = pd.DataFrame([{
            "Employee Name": employee_name,
            "Probability": round(probability, 4),
            "Risk Level": label,
            "Priority": ai_result["priority"],
            "HR Owner": hr_owner,
            "Status": "Pending"
        }])
        tracker_df = pd.concat([tracker_df, new_row], ignore_index=True)
        tracker_df = tracker_df.drop_duplicates(subset=["Employee Name"], keep="last")
        save_action_tracker(tracker_df)

        history_df = load_prediction_history()
        history_row = pd.DataFrame([{
            "Employee Name": employee_name,
            "Probability": round(probability, 4),
            "Risk Level": label,
            "Priority": ai_result["priority"],
            "HR Owner": hr_owner,
            "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }])
        history_df = pd.concat([history_df, history_row], ignore_index=True)
        save_prediction_history(history_df)

    st.subheader("Optional Email Alert to HR")
    st.markdown('<div class="email-box">', unsafe_allow_html=True)
    sender_email = st.text_input("Sender Gmail")
    sender_password = st.text_input("App Password", type="password")
    receiver_email = st.text_input("HR Email")

    if st.button("Send Email Alert to HR"):
        if st.session_state.last_prediction is None:
            st.warning("Please run a prediction first.")
        elif not sender_email or not sender_password or not receiver_email:
            st.warning("Please fill all email fields.")
        else:
            subject = f"Employee Attrition Alert - {st.session_state.last_prediction['employee_name']}"
            success, message = send_email_alert(
                sender_email=sender_email,
                sender_password=sender_password,
                receiver_email=receiver_email,
                subject=subject,
                body=st.session_state.last_prediction["hr_notice"]
            )
            if success:
                st.success(message)
            else:
                st.error(message)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------
# PAGE 7: PREDICTION HISTORY
# ---------------------------------
elif menu == "Prediction History":
    st.header("Prediction History")

    history_df = load_prediction_history()

    if history_df.empty:
        st.info("No prediction history available yet.")
    else:
        st.subheader("Saved Predictions")
        st.dataframe(history_df, use_container_width=True)

        st.download_button(
            label="Download Prediction History",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv"
        )

# ---------------------------------
# PAGE 8: BULK PREDICTION
# ---------------------------------
elif menu == "Bulk Prediction":
    st.header("Bulk CSV Prediction")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        bulk_df_raw = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data Preview")
        st.dataframe(bulk_df_raw.head(), use_container_width=True)

        try:
            original_bulk_df = bulk_df_raw.copy()
            bulk_df = preprocess_bulk_data(bulk_df_raw.copy(), model_columns)

            predictions = model.predict(bulk_df)
            probabilities = model.predict_proba(bulk_df)[:, 1]

            original_bulk_df["Predicted_Attrition"] = predictions
            original_bulk_df["Attrition_Probability"] = probabilities
            original_bulk_df["Risk_Level"] = original_bulk_df["Attrition_Probability"].apply(risk_label)

            def bulk_recommendation(row):
                recs = []
                if row.get("OverTime", "") == "Yes":
                    recs.append("Reduce overtime")
                if row.get("JobSatisfaction", 4) <= 2:
                    recs.append("Improve job satisfaction")
                if row.get("MonthlyIncome", 99999) < 4000:
                    recs.append("Review salary")
                if row.get("WorkLifeBalance", 4) <= 2:
                    recs.append("Improve work-life balance")
                if row.get("TrainingTimesLastYear", 5) <= 1:
                    recs.append("Provide training")
                if not recs:
                    recs.append("Maintain engagement")
                return ", ".join(recs)

            original_bulk_df["Suggested_Action"] = original_bulk_df.apply(bulk_recommendation, axis=1)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Employees", len(original_bulk_df))
            c2.metric("Low Risk", int((original_bulk_df["Risk_Level"] == "Low Risk").sum()))
            c3.metric("Medium Risk", int((original_bulk_df["Risk_Level"] == "Medium Risk").sum()))
            c4.metric("High Risk", int((original_bulk_df["Risk_Level"] == "High Risk").sum()))

            st.subheader("Risk Summary")
            risk_counts = original_bulk_df["Risk_Level"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            risk_counts.plot(kind="bar", ax=ax)
            ax.set_title("Bulk Risk Distribution")
            ax.set_xlabel("Risk Level")
            ax.set_ylabel("Count")
            plt.xticks(rotation=0)
            st.pyplot(fig)

            st.subheader("Top High-Risk Employees")
            high_risk_df = original_bulk_df.sort_values(by="Attrition_Probability", ascending=False).head(10)
            st.dataframe(high_risk_df, use_container_width=True)

            selected_risk = st.selectbox("Filter by Risk Level", ["All", "High Risk", "Medium Risk", "Low Risk"])
            filtered_df = original_bulk_df.copy()
            if selected_risk != "All":
                filtered_df = filtered_df[filtered_df["Risk_Level"] == selected_risk]

            st.subheader("Prediction Results")
            st.dataframe(filtered_df, use_container_width=True)

            st.download_button(
                label="Download Filtered Prediction Results",
                data=filtered_df.to_csv(index=False).encode("utf-8"),
                file_name="bulk_attrition_predictions.csv",
                mime="text/csv"
            )

            st.download_button(
                label="Download High-Risk Employees Only",
                data=original_bulk_df[original_bulk_df["Risk_Level"] == "High Risk"].to_csv(index=False).encode("utf-8"),
                file_name="high_risk_employees.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Bulk prediction failed: {e}")

# ---------------------------------
# PAGE 9: HR ALERT CENTER
# ---------------------------------
elif menu == "HR Alert Center":
    st.header("HR Alert Center")

    uploaded_file = st.file_uploader("Upload Employee CSV for HR Alerts", type=["csv"], key="hr_alert_csv")

    if uploaded_file is not None:
        bulk_df_raw = pd.read_csv(uploaded_file)

        try:
            original_bulk_df = bulk_df_raw.copy()
            bulk_df = preprocess_bulk_data(bulk_df_raw.copy(), model_columns)

            probabilities = model.predict_proba(bulk_df)[:, 1]
            predictions = model.predict(bulk_df)

            original_bulk_df["Predicted_Attrition"] = predictions
            original_bulk_df["Attrition_Probability"] = probabilities
            original_bulk_df["Risk_Level"] = original_bulk_df["Attrition_Probability"].apply(risk_label)
            original_bulk_df["Priority"] = original_bulk_df["Attrition_Probability"].apply(priority_label)
            original_bulk_df["HR_Owner"] = original_bulk_df["Risk_Level"].apply(assign_hr_owner)

            def build_alert(row):
                overtime_val = row.get("OverTime", "No")
                job_sat = row.get("JobSatisfaction", 4)
                income = row.get("MonthlyIncome", 5000)
                work_life = row.get("WorkLifeBalance", 4)
                training = row.get("TrainingTimesLastYear", 2)
                promotion_gap = row.get("YearsSinceLastPromotion", 0)
                env_sat = row.get("EnvironmentSatisfaction", 4)
                dist = row.get("DistanceFromHome", 5)
                companies = row.get("NumCompaniesWorked", 1)

                ai_result = generate_ai_recommendations(
                    probability=row["Attrition_Probability"],
                    overtime=overtime_val,
                    job_satisfaction=job_sat,
                    monthly_income=income,
                    work_life_balance=work_life,
                    training_times_last_year=training,
                    years_since_last_promotion=promotion_gap,
                    environment_satisfaction=env_sat,
                    distance_from_home=dist,
                    num_companies_worked=companies
                )

                emp_name = row.get("EmployeeName", f"Employee_{row.name + 1}")
                all_recommendations = ai_result["hr_actions"] + ai_result["manager_actions"]

                return generate_hr_notice(
                    employee_name=emp_name,
                    probability=row["Attrition_Probability"],
                    risk_level=row["Risk_Level"],
                    issues=ai_result["issues"],
                    recommendations=all_recommendations
                )

            original_bulk_df["Escalation_Status"] = original_bulk_df.apply(
                lambda row: escalation_flag(
                    row["Attrition_Probability"],
                    row.get("OverTime", "No"),
                    row.get("JobSatisfaction", 4),
                    row.get("WorkLifeBalance", 4),
                ),
                axis=1
            )

            original_bulk_df["HR_Notice"] = original_bulk_df.apply(build_alert, axis=1)

            attention_df = original_bulk_df[
                original_bulk_df["Risk_Level"].isin(["High Risk", "Medium Risk"])
            ].sort_values(by="Attrition_Probability", ascending=False)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Alerts", len(attention_df))
            c2.metric("High Risk Alerts", int((attention_df["Risk_Level"] == "High Risk").sum()))
            c3.metric("Critical Priority", int((attention_df["Priority"] == "Critical").sum()))
            c4.metric("Escalate Immediately", int((attention_df["Escalation_Status"] == "Escalate Immediately").sum()))

            st.subheader("Critical Employee Watchlist")
            watchlist_cols = [col for col in [
                "EmployeeName", "Attrition_Probability", "Risk_Level",
                "Priority", "HR_Owner", "Escalation_Status"
            ] if col in attention_df.columns]

            if not watchlist_cols:
                watchlist_cols = ["Attrition_Probability", "Risk_Level", "Priority", "HR_Owner", "Escalation_Status"]

            st.dataframe(attention_df[watchlist_cols], use_container_width=True)

            if not attention_df.empty:
                st.subheader("Generated HR Notices")
                selected_index = st.selectbox("Select employee row to view HR notice", attention_df.index.tolist())
                selected_row = attention_df.loc[selected_index]
                st.text_area("Generated HR Notice", selected_row["HR_Notice"], height=280)

                st.download_button(
                    label="Download Selected HR Notice",
                    data=selected_row["HR_Notice"],
                    file_name="selected_hr_notice.txt",
                    mime="text/plain"
                )

            st.download_button(
                label="Download HR Alert Report",
                data=attention_df.to_csv(index=False).encode("utf-8"),
                file_name="hr_alert_report.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"HR alert generation failed: {e}")

# ---------------------------------
# PAGE 10: HR ACTION TRACKER
# ---------------------------------
elif menu == "HR Action Tracker":
    st.header("HR Action Tracker")

    tracker_df = load_action_tracker()

    if tracker_df.empty:
        st.info("No HR actions recorded yet.")
    else:
        st.subheader("Current Tracker")
        st.dataframe(tracker_df, use_container_width=True)

        st.subheader("Update Employee Status")
        selected_employee = st.selectbox("Select Employee", tracker_df["Employee Name"].tolist())
        new_status = st.selectbox("Select New Status", ["Pending", "Reviewed", "Action Taken", "Escalated", "Retained"])

        if st.button("Update Status"):
            tracker_df.loc[tracker_df["Employee Name"] == selected_employee, "Status"] = new_status
            save_action_tracker(tracker_df)
            st.success("Status updated successfully.")
            st.dataframe(tracker_df, use_container_width=True)

        st.download_button(
            label="Download HR Action Tracker",
            data=tracker_df.to_csv(index=False).encode("utf-8"),
            file_name="hr_action_tracker.csv",
            mime="text/csv"
        )

# ---------------------------------
# PAGE 11: AI RECOMMENDATIONS
# ---------------------------------
elif menu == "AI Recommendations":
    st.header("AI-Based Recommendation Engine")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Compensation Actions")
        st.write("- Review salary for low-income profiles")
        st.write("- Improve benefits and incentives")

        st.subheader("Career Growth Actions")
        st.write("- Create promotion planning for stagnating employees")
        st.write("- Offer internal mobility and skill growth")

    with col2:
        st.subheader("Workload & Wellbeing Actions")
        st.write("- Reduce overtime burden")
        st.write("- Improve work-life balance support")
        st.write("- Monitor burnout indicators")

        st.subheader("Managerial & Workplace Support")
        st.write("- Improve team culture and manager support")
        st.write("- Increase engagement and recognition")

    st.info("This makes the system stronger because it not only predicts attrition, but also recommends preventive actions for HR teams.")

# ---------------------------------
# PAGE 12: HR CHATBOT
# ---------------------------------
elif menu == "HR Chatbot":
    st.header("HR Chatbot Assistant")
    st.write("Ask questions about attrition, HR actions, dashboard, SHAP, or employee risk.")

    user_query = st.text_input("Type your question here")

    if st.button("Ask Chatbot"):
        if user_query.strip():
            bot_reply = chatbot_response(user_query)
            st.session_state.chat_history.append(("You", user_query))
            st.session_state.chat_history.append(("Bot", bot_reply))

    if st.session_state.chat_history:
        st.subheader("Chat History")
        for sender, message in reversed(st.session_state.chat_history):
            if sender == "You":
                st.markdown(f"**🧑 You:** {message}")
            else:
                st.markdown(f"**🤖 Bot:** {message}")

# ---------------------------------
# PAGE 13: FEATURE IMPORTANCE
# ---------------------------------
elif menu == "Feature Importance":
    st.header("Feature Importance")

    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature": model_columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(importance_df["Feature"], importance_df["Importance"])
        ax.invert_yaxis()
        ax.set_title("Top 15 Important Features")
        ax.set_xlabel("Importance Score")
        st.pyplot(fig)

        st.dataframe(importance_df, use_container_width=True)
    else:
        st.warning("Feature importance not available for this model.")

# ---------------------------------
# PAGE 14: SHAP EXPLAINABILITY
# ---------------------------------
elif menu == "SHAP Explainability":
    st.header("SHAP Explainability")
    st.write("This page explains why the model predicted attrition risk for an employee.")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 60, 30, key="shap_age")
        monthly_income = st.number_input("Monthly Income", 1000, 50000, 5000, key="shap_income")
        job_satisfaction = st.slider("Job Satisfaction", 1, 4, 2, key="shap_job_sat")
        work_life_balance = st.slider("Work Life Balance", 1, 4, 2, key="shap_wlb")

    with col2:
        overtime = st.selectbox("OverTime", ["Yes", "No"], key="shap_overtime")
        years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 1, key="shap_promo")
        distance_from_home = st.slider("Distance From Home", 1, 30, 5, key="shap_dist")
        num_companies_worked = st.slider("Num Companies Worked", 0, 10, 2, key="shap_companies")

    if st.button("Explain Prediction"):
        try:
            import shap
        except Exception:
            st.error("SHAP is not installed. Run: pip install shap")
            st.stop()

        input_dict = {
            "Age": age,
            "MonthlyIncome": monthly_income,
            "JobSatisfaction": job_satisfaction,
            "WorkLifeBalance": work_life_balance,
            "YearsSinceLastPromotion": years_since_last_promotion,
            "DistanceFromHome": distance_from_home,
            "NumCompaniesWorked": num_companies_worked,
            "OverTime_Yes": 1 if overtime == "Yes" else 0
        }

        input_df = pd.DataFrame([input_dict])

        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[model_columns]
        prob = model.predict_proba(input_df)[0][1]
        st.metric("Attrition Probability", f"{prob:.2%}")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        if isinstance(shap_values, list):
            shap_array = shap_values[1][0]
        else:
            shap_array = shap_values[0]

        shap_df = pd.DataFrame({
            "Feature": model_columns,
            "SHAP Value": shap_array
        })

        shap_df["Abs_SHAP"] = shap_df["SHAP Value"].abs()
        shap_df = shap_df.sort_values(by="Abs_SHAP", ascending=False).head(10)

        st.subheader("Top Features Affecting Prediction")
        st.dataframe(shap_df[["Feature", "SHAP Value"]], use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(shap_df["Feature"], shap_df["SHAP Value"])
        ax.invert_yaxis()
        ax.set_title("Top SHAP Feature Contributions")
        ax.set_xlabel("SHAP Value")
        st.pyplot(fig)

# ---------------------------------
# END CLEAN
# ---------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)