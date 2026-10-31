import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

st.markdown("""
<style>
/* Keep the pastel app readable even when Streamlit's global theme is dark. */
:root { color-scheme: light; }
.stApp, [data-testid="stAppViewContainer"] {
    background:linear-gradient(135deg,#fff8fc 0%,#f4f1ff 50%,#effaff 100%) !important;
    color:#303650 !important;
}
[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#eee7ff,#e4f5ff) !important;
}
.block-container {max-width:1180px;padding-top:2rem}
.stApp p, .stApp li, .stApp label, .stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
    color:#303650 !important;
}
h1,h2,h3,h4,h5,h6 {color:#343653!important}
.hero {background:linear-gradient(110deg,#e8ddff,#dff4ff 58%,#ffe8ef);padding:28px 32px;border-radius:24px;margin-bottom:22px;border:1px solid #e5e0f4}
.hero h1 {margin:0;color:#343653!important;font-size:2.3rem}
.hero p {margin:8px 0 0;color:#59617b!important;font-size:1.05rem}
.pill {display:inline-block;background:white;color:#7462ad!important;border-radius:999px;padding:5px 12px;font-size:.82rem;font-weight:700;margin-bottom:10px}
div[data-testid="stMetric"] {background:white;border:1px solid #e8e5f5;padding:18px 20px;border-radius:18px;box-shadow:0 5px 18px #4c488214}
div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {color:#303650!important}
div[data-testid="stForm"] {background:#ffffffdf;padding:24px;border:1px solid #e8e5f5;border-radius:20px;box-shadow:0 6px 22px #4c488210}
.soft-card {background:white;border:1px solid #e8e5f5;border-radius:18px;padding:18px 20px;margin:8px 0;color:#303650}
.suggestion {background:white;border-left:5px solid #b9a8ef;border-radius:12px;padding:13px 16px;margin:9px 0;color:#303650}
.stApp input, .stApp textarea, .stApp [data-baseweb="select"] > div {
    background-color:#ffffff!important;
    color:#303650!important;
}
.stApp input, .stApp textarea { -webkit-text-fill-color:#303650!important; }
.stApp [data-baseweb="select"] * {color:#303650!important;}
div.stButton>button,div[data-testid="stFormSubmitButton"] button {background:linear-gradient(90deg,#9d8be8,#79bfe8);color:white!important;border:0;border-radius:12px;font-weight:700}

/* Strong contrast overrides for Streamlit dark-theme widgets and sidebar. */
[data-testid="stSidebar"], [data-testid="stSidebar"] * {
    color: #303650 !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #303650 !important;
}
[data-testid="stAlert"], [data-testid="stAlert"] * {
    color: #303650 !important;
}
[data-testid="stMetric"], [data-testid="stMetric"] * {
    color: #303650 !important;
}
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *,
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] * {
    color: #303650 !important;
}
.stApp [data-baseweb="select"] *,
.stApp [data-baseweb="input"] *,
.stApp [data-baseweb="textarea"] * {
    color: #303650 !important;
    -webkit-text-fill-color: #303650 !important;
}
.stApp input, .stApp textarea {
    caret-color: #303650 !important;
}
.stApp div[data-testid="stForm"] *,
.stApp div[data-testid="stExpander"] * {
    color: #303650 !important;
}
.stApp button, .stApp button * {
    color: #ffffff !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<span class="pill">✨ YOUR LEARNING COMPANION</span>
<h1>🎓 Student Performance Predictor</h1>
<p>Explore how everyday habits relate to exam performance and get practical ideas to improve your routine.</p>
</div>
""", unsafe_allow_html=True)

st.info("This is an educational estimate based on patterns in the training dataset. It is not a guaranteed result or a measure of a student's ability.")

DATA_FILE = "studenthabits.csv"
REQUIRED_COLUMNS = [
    "age", "gender", "study_hours_per_day", "social_media_hours",
    "attendance_percentage", "sleep_hours", "diet_quality",
    "exercise_frequency", "internet_quality",
    "extracurricular_participation", "exam_score"
]

@st.cache_resource
def train_model(file_path: str):
    data = pd.read_csv(file_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError("The dataset is missing these required columns: " + ", ".join(missing))
    data = data[REQUIRED_COLUMNS].copy()
    data["exam_score"] = pd.to_numeric(data["exam_score"], errors="coerce")
    data = data.dropna(subset=["exam_score"])
    if len(data) < 5:
        raise ValueError("The dataset needs at least 5 rows with valid exam scores.")
    X = data.drop(columns=["exam_score"])
    y = data["exam_score"]
    num_cols = X.select_dtypes(include="number").columns
    cat_cols = X.select_dtypes(exclude="number").columns
    num_process = SimpleImputer(strategy="median")
    cat_process = Pipeline([
        ("fill", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocess = ColumnTransformer([
        ("num", num_process, num_cols),
        ("cat", cat_process, cat_cols)
    ])
    model = Pipeline([
        ("preprocess", preprocess),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    test_predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, test_predictions)
    model.fit(X, y)
    return model, mae, len(data)

def get_suggestions(student):
    suggestions = []
    if student["study_hours_per_day"] < 2:
        suggestions.append("Create a regular study timetable and build up focused study time gradually.")
    if student["social_media_hours"] > 3:
        suggestions.append("Set limits on social media, especially during planned study sessions.")
    if student["attendance_percentage"] < 75:
        suggestions.append("Work toward attending at least 75% of classes, and ask for help with missed topics.")
    if student["sleep_hours"] < 7:
        suggestions.append("Aim for a consistent sleep routine and enough rest.")
    if student["diet_quality"] == "Poor":
        suggestions.append("Try to eat regular, balanced meals.")
    if student["exercise_frequency"] < 3:
        suggestions.append("Add enjoyable physical activity to the week.")
    if student["internet_quality"] == "Poor":
        suggestions.append("Download notes and learning resources in advance so you can study offline.")
    if student["extracurricular_participation"] == "No":
        suggestions.append("Consider an extracurricular activity that interests you, while keeping study time balanced.")
    if not suggestions:
        suggestions.append("Keep your routine consistent and review your progress regularly.")
    return suggestions

with st.sidebar:
    st.markdown("## 🌷 About this app")
    st.write("The model learns patterns from `studenthabits.csv`.")
    st.write("**AI model:** Random Forest Regression")
    st.markdown("---")
    st.markdown("### 💡 How to use")
    st.write("1. Enter student habits.\n2. Predict the score.\n3. Review suggestions and charts.")
    st.caption("Keep the CSV file in the same folder as this Python file.")

try:
    model, mae, row_count = train_model(DATA_FILE)
except FileNotFoundError:
    st.error(f"Could not find `{DATA_FILE}`. Put studenthabits.csv in the same folder as this Python file, then restart the app.")
    st.stop()
except Exception as exc:
    st.error(f"Could not prepare the model: {exc}")
    st.stop()

status1, status2 = st.columns([2, 1])
with status1:
    st.success(f"Model ready · trained using {row_count:,} dataset rows")
with status2:
    st.metric("Test-set mean absolute error", f"{mae:.2f} marks")
st.caption("MAE is the average absolute difference between predictions and actual scores in the held-out test data.")

st.markdown("---")
st.markdown("## 📝 Student details")
st.write("Fill in the details below, then select **Predict my score**.")

with st.form("student_details"):
    st.markdown("### 👤 Basic information")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=5, max_value=100, value=18, step=1)
    with c2:
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    with c3:
        attendance = st.slider("Attendance percentage", 0, 100, 80)

    st.markdown("### 📚 Daily routine")
    c1, c2, c3 = st.columns(3)
    with c1:
        study_hours = st.number_input("Study hours per day", 0.0, 24.0, 2.0, 0.5)
    with c2:
        social_hours = st.number_input("Social media hours per day", 0.0, 24.0, 2.0, 0.5)
    with c3:
        sleep = st.number_input("Sleep hours per day", 0.0, 24.0, 8.0, 0.5)

    st.markdown("### 🌱 Lifestyle")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        diet = st.selectbox("Diet quality", ["Good", "Fair", "Poor"])
    with c2:
        exercise = st.slider("Exercise days per week", 0, 7, 3)
    with c3:
        internet = st.selectbox("Internet quality", ["Good", "Average", "Poor"])
    with c4:
        extracurricular = st.selectbox("Extracurriculars", ["Yes", "No"])

    submitted = st.form_submit_button("✨ Predict my score", type="primary", use_container_width=True)

if submitted:
    student = {
        "age": int(age), "gender": gender,
        "study_hours_per_day": float(study_hours),
        "social_media_hours": float(social_hours),
        "attendance_percentage": float(attendance),
        "sleep_hours": float(sleep), "diet_quality": diet,
        "exercise_frequency": int(exercise), "internet_quality": internet,
        "extracurricular_participation": extracurricular
    }
    student_df = pd.DataFrame([student])
    predicted_score = float(model.predict(student_df)[0])
    predicted_score = max(0.0, min(100.0, predicted_score))
    suggestions = get_suggestions(student)

    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    st.session_state.prediction_history.insert(0, {"score": predicted_score})
    st.session_state.prediction_history = st.session_state.prediction_history[:10]

    st.markdown("---")
    st.markdown("## 🌟 Your student report")
    score_col, detail_col = st.columns([1, 1.25])
    with score_col:
        st.metric("Predicted exam score", f"{predicted_score:.1f} / 100")
        st.progress(int(round(predicted_score)))
        st.caption("Estimated score based on the habits entered.")
    with detail_col:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown("### 💜 Model estimate")
        st.write("The estimate is generated from the habits entered and relationships learned from the dataset.")
        st.write(f"Test-set mean absolute error: **{mae:.2f} marks**")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 🌈 Personalised suggestions")
    for suggestion in suggestions:
        st.markdown(f'<div class="suggestion">💡 {suggestion}</div>', unsafe_allow_html=True)

    st.markdown("### 📊 Your daily habits")
    chart_data = pd.DataFrame({
        "Habit": ["Study", "Social media", "Sleep"],
        "Hours per day": [study_hours, social_hours, sleep]
    }).set_index("Habit")
    st.bar_chart(chart_data, y="Hours per day")

    with st.expander("👀 View entered student details"):
        st.dataframe(student_df, use_container_width=True)

    report_text = (
        "STUDENT PERFORMANCE PREDICTOR REPORT\n"
        "====================================\n"
        f"Predicted exam score: {predicted_score:.1f}/100\n\n"
        "STUDENT DETAILS\n" + student_df.to_string(index=False) +
        "\n\nSUGGESTIONS\n" + "\n".join(f"- {item}" for item in suggestions) +
        "\n\nEducational estimate only; not a guaranteed result.\n"
    )
    st.download_button("⬇️ Download student report", report_text,
                       file_name="student_performance_report.txt", mime="text/plain")

if "prediction_history" in st.session_state and st.session_state.prediction_history:
    st.markdown("---")
    st.markdown("## 🕘 Recent predictions")
    history = [{"Prediction": i + 1, "Estimated score": f"{x['score']:.1f}/100"}
               for i, x in enumerate(st.session_state.prediction_history)]
    st.dataframe(pd.DataFrame(history), hide_index=True, use_container_width=True)

st.markdown("---")
st.caption("🎓 Student Performance Predictor · Educational project")
