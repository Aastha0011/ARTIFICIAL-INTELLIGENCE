import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Employee Retention Prediction",
    page_icon="👨‍💼",
    layout="centered"
)

st.title("👨‍💼 Employee Retention Prediction")
st.write("Predict whether an employee is likely to leave the company.")

# Load dataset
df = pd.read_csv("PROJECT 4 retention project/HR_comma_sep.csv")

# Feature Engineering
subdf = df[['satisfaction_level',
            'average_montly_hours',
            'promotion_last_5years',
            'salary']]

salary_dummies = pd.get_dummies(subdf['salary'], prefix='salary')

df_with_dummies = pd.concat([subdf, salary_dummies], axis=1)
df_with_dummies.drop('salary', axis=1, inplace=True)

X = df_with_dummies
y = df['left']

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

st.sidebar.header("Model Accuracy")
st.sidebar.success(f"{accuracy*100:.2f}%")

st.header("Enter Employee Details")

satisfaction = st.slider(
    "Satisfaction Level",
    0.0,
    1.0,
    0.5,
    0.01
)

hours = st.slider(
    "Average Monthly Hours",
    50,
    350,
    200
)

promotion = st.selectbox(
    "Promotion in Last 5 Years",
    [0, 1]
)

salary = st.selectbox(
    "Salary Level",
    ["low", "medium", "high"]
)

salary_low = 1 if salary == "low" else 0
salary_medium = 1 if salary == "medium" else 0
salary_high = 1 if salary == "high" else 0

input_data = pd.DataFrame([[
    satisfaction,
    hours,
    promotion,
    salary_high,
    salary_low,
    salary_medium
]], columns=X.columns)

if st.button("Predict"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.error("⚠️ This employee is likely to leave the company.")
    else:
        st.success("✅ This employee is likely to stay with the company.")

    st.write(f"**Probability of Staying:** {probability[0]*100:.2f}%")
    st.write(f"**Probability of Leaving:** {probability[1]*100:.2f}%")
