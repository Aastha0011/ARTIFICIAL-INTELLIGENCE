import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Canada Income Predictor")

st.title("🇨🇦 Canada Per Capita Income Predictor")

st.write("Predict Canada's Per Capita Income using Linear Regression.")

# Load data
df = pd.read_csv("canada_per_capita_income.csv")

# Train model
X = df[['year']]
y = df['per capita income (US$)']

model = LinearRegression()
model.fit(X, y)

# User input
year = st.number_input(
    "Enter Year",
    min_value=1960,
    max_value=2050,
    value=2020,
    step=1
)

# Prediction
if st.button("Predict"):
    prediction = model.predict([[year]])
    st.success(f"Predicted Income: ${prediction[0]:,.2f}")

# Show dataset
if st.checkbox("Show Dataset"):
    st.dataframe(df)
