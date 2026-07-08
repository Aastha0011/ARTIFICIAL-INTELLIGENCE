import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="K-Means Clustering", layout="wide")

st.title("K-Means Clustering on Income Dataset")
st.write("Upload the income.csv file to visualize customer clusters.")

uploaded_file = st.file_uploader("Upload income.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset")
    st.dataframe(df)

    # Scaling
    scaler_age = MinMaxScaler()
    scaler_income = MinMaxScaler()

    df["Age"] = scaler_age.fit_transform(df[["Age"]])
    df["Income($)"] = scaler_income.fit_transform(df[["Income($)"]])

    st.subheader("Choose Number of Clusters")
    k = st.slider("Number of Clusters", 2, 10, 3)

    # Train model
    km = KMeans(n_clusters=k, random_state=42)
    df["Cluster"] = km.fit_predict(df[["Age", "Income($)"]])

    st.subheader("Clustered Data")
    st.dataframe(df)

    # Plot clusters
    fig, ax = plt.subplots(figsize=(8,6))

    for cluster in range(k):
        cluster_data = df[df["Cluster"] == cluster]
        ax.scatter(
            cluster_data["Age"],
            cluster_data["Income($)"],
            label=f"Cluster {cluster}"
        )

    centers = km.cluster_centers_
    ax.scatter(
        centers[:,0],
        centers[:,1],
        marker="*",
        s=250,
        color="black",
        label="Centroids"
    )

    ax.set_xlabel("Age (Scaled)")
    ax.set_ylabel("Income (Scaled)")
    ax.legend()

    st.pyplot(fig)

    # Elbow Method
    st.subheader("Elbow Method")

    sse = []
    K = range(1,11)

    for i in K:
        model = KMeans(n_clusters=i, random_state=42)
        model.fit(df[["Age","Income($)"]])
        sse.append(model.inertia_)

    fig2, ax2 = plt.subplots(figsize=(8,5))
    ax2.plot(K, sse, marker="o")
    ax2.set_xlabel("Number of Clusters")
    ax2.set_ylabel("SSE")
    ax2.set_title("Elbow Method")

    st.pyplot(fig2)

else:
    st.info("Please upload income.csv to continue.")
