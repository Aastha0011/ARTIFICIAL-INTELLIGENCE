import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

st.set_page_config(page_title="K-Means Iris Clustering", page_icon="🌸")

st.title("🌸 Iris Flower Cluster Prediction")
st.write("Enter the petal measurements to predict the cluster using K-Means.")

# Load Iris dataset
iris = load_iris()

# Create dataframe
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Keep only the required columns
df = df[['petal length (cm)', 'petal width (cm)']]

# Train KMeans model
model = KMeans(n_clusters=3, random_state=42)
model.fit(df)

# User Inputs
petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=4.5,
    step=0.1
)

petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    max_value=5.0,
    value=1.5,
    step=0.1
)

if st.button("Predict Cluster"):

    sample = [[petal_length, petal_width]]

    prediction = model.predict(sample)[0]

    st.success(f"Predicted Cluster: {prediction}")

    if prediction == 0:
        st.info("This flower belongs to Cluster 0.")
    elif prediction == 1:
        st.info("This flower belongs to Cluster 1.")
    else:
        st.info("This flower belongs to Cluster 2.")
