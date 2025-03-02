import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import math

st.title("Distance Metrics Demo with Make Moons")
st.write("""
Distance metrics measure how "different" two data points are. This demo uses the 'make_moons' dataset to show how various distance metrics (Euclidean, Manhattan, Minkowski, Cosine) evaluate similarity.
""")

# Sidebar for user inputs
metric = st.sidebar.selectbox("Select a distance metric:", ["Euclidean", "Manhattan", "Minkowski", "Cosine"])

if metric == "Minkowski":
    minkowski_power = st.sidebar.slider("Minkowski Power", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
else:
    minkowski_power = 3.0

n_samples = st.sidebar.slider("Number of samples", 100, 1000, 200, step=50)
noise = st.sidebar.slider("Noise level", 0.0, 0.5, 0.1, step=0.05)

# Generate the make_moons dataset
X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
df = pd.DataFrame(X, columns=["x", "y"])
df["label"] = y

st.write("### Generated 'Make Moons' Dataset")
st.dataframe(df.head())

# Plot the dataset
fig, ax = plt.subplots()
scatter = ax.scatter(df["x"], df["y"], c=df["label"], cmap="viridis")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Make Moons Dataset")
st.pyplot(fig)

# Define distance functions
def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def minkowski(p1, p2, power):
    return ((abs(p1[0] - p2[0])**power + abs(p1[1] - p2[1])**power))**(1.0/power)

def cosine_distance(p1, p2):
    dot = p1[0]*p2[0] + p1[1]*p2[1]
    mag1 = math.sqrt(p1[0]**2 + p1[1]**2)
    mag2 = math.sqrt(p2[0]**2 + p2[1]**2)
    if mag1 == 0 or mag2 == 0:
        return 1
    cos_sim = dot / (mag1 * mag2)
    return 1 - cos_sim

# Map metric selection to the corresponding function
if metric == "Euclidean":
    distance_func = euclidean
elif metric == "Manhattan":
    distance_func = manhattan
elif metric == "Minkowski":
    distance_func = lambda p1, p2: minkowski(p1, p2, minkowski_power)
elif metric == "Cosine":
    distance_func = cosine_distance

# Allow user to select a point index from which to compute distances
selected_index = st.sidebar.slider("Select a point index", 0, len(X)-1, 0)
selected_point = X[selected_index]
st.write(f"### Distances from point {selected_index}: {selected_point}")

# Compute distances from the selected point to all other points
distances = [distance_func(selected_point, X[i]) for i in range(len(X))]
df_distances = pd.DataFrame({"Index": range(len(X)), "Distance": distances})
df_distances = df_distances.sort_values("Distance")

st.write("### Nearest Neighbors")
st.dataframe(df_distances.head(10))

# Highlight the selected point and its nearest neighbor on the scatter plot
fig2, ax2 = plt.subplots()
scatter = ax2.scatter(df["x"], df["y"], c=df["label"], cmap="viridis", label="Data Points")
ax2.scatter(selected_point[0], selected_point[1], color="red", s=100, label="Selected Point")
# The closest point is the first one (itself) so take the second closest as the nearest neighbor
nearest_index = df_distances.iloc[1]["Index"]
nearest_point = X[int(nearest_index)]
ax2.scatter(nearest_point[0], nearest_point[1], color="orange", s=100, label="Nearest Neighbor")
ax2.legend()
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title(f"Nearest Neighbor using {metric} Distance")
st.pyplot(fig2)
