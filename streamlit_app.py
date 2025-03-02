import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import math

# -----------------------------------------------------------------------------
# Title and Introduction
# -----------------------------------------------------------------------------
st.title("Distance Metrics Demo with Make Moons")
st.write("""
Distance metrics might sound like an abstract technical concept, but they lie at the heart of many machine learning processes. They help an algorithm decide whether two pieces of data are close or far apart in terms of similarity.

In this demo, we use the 'make_moons' dataset – a toy dataset that creates two interleaving crescent shapes – to show how different distance metrics (like Euclidean, Manhattan, Minkowski, and Cosine) measure the 'difference' between points. Think of it as a way to decide which data points are like friends at a dinner party!
""")

# -----------------------------------------------------------------------------
# Sidebar: Explanation of Make Moons and Hyperparameters
# -----------------------------------------------------------------------------
st.sidebar.header("About the Dataset & Hyperparameters")

st.sidebar.write("""
**Make Moons Dataset:**  
The 'make_moons' dataset is a synthetic, two-dimensional dataset that forms two crescent shapes (like two moons). It is useful for demonstrating clustering and classification because the clusters are not perfectly circular – they curve around each other.

**Hyperparameters:**
- **Number of Samples:** Controls how many data points are generated.
- **Noise Level:** Adds randomness to the data. A higher noise level means the data points will be more spread out and less cleanly separated.
""")

# Distance Metric selection
metric = st.sidebar.selectbox("Select a distance metric:", 
                              ["Euclidean", "Manhattan", "Minkowski", "Cosine"])

# For Minkowski, let the user adjust the power parameter
if metric == "Minkowski":
    minkowski_power = st.sidebar.slider("Minkowski Power", min_value=1.0, max_value=5.0, value=3.0, step=0.1,
                                        help="This parameter controls the 'dial' between Manhattan (power=1) and Euclidean (power=2) behaviour, or even beyond.")
else:
    minkowski_power = 3.0  # Default value (unused for other metrics)

# Dataset hyperparameters
n_samples = st.sidebar.slider("Number of Samples", 100, 1000, 200, step=50,
                              help="Choose how many points to generate in the dataset.")
noise = st.sidebar.slider("Noise Level", 0.0, 0.5, 0.1, step=0.05,
                           help="Adjust the noise level to see how 'messy' the data becomes.")

# -----------------------------------------------------------------------------
# Generate the Make Moons Dataset
# -----------------------------------------------------------------------------
X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
df = pd.DataFrame(X, columns=["x", "y"])
df["label"] = y

st.write("### Generated 'Make Moons' Dataset")
st.write("The dataset consists of two interleaving crescent shapes. Each point has two features (x and y coordinates) and a label (0 or 1) indicating which moon it belongs to.")
st.dataframe(df.head(10))

# -----------------------------------------------------------------------------
# Plot the Dataset
# -----------------------------------------------------------------------------
fig, ax = plt.subplots()
scatter = ax.scatter(df["x"], df["y"], c=df["label"], cmap="viridis")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Make Moons Dataset")
st.pyplot(fig)

st.write("""
The scatter plot above shows the 'make_moons' dataset. Notice the two curved clusters. 
Each point is coloured by its label (which moon it belongs to).
""")

# -----------------------------------------------------------------------------
# Define Distance Functions
# -----------------------------------------------------------------------------
def euclidean(p1, p2):
    """Calculate the straight-line (Euclidean) distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def manhattan(p1, p2):
    """Calculate the Manhattan distance (sum of absolute differences)."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def minkowski(p1, p2, power):
    """Calculate the Minkowski distance with a given power parameter."""
    return ((abs(p1[0] - p2[0])**power + abs(p1[1] - p2[1])**power))**(1.0/power)

def cosine_distance(p1, p2):
    """Calculate the Cosine distance, which is 1 minus the cosine similarity."""
    dot = p1[0]*p2[0] + p1[1]*p2[1]
    mag1 = math.sqrt(p1[0]**2 + p1[1]**2)
    mag2 = math.sqrt(p2[0]**2 + p2[1]**2)
    if mag1 == 0 or mag2 == 0:
        return 1  # If either vector is zero-length, treat as maximally distant.
    cos_sim = dot / (mag1 * mag2)
    return 1 - cos_sim

# Map the metric selection to the corresponding function
if metric == "Euclidean":
    distance_func = euclidean
elif metric == "Manhattan":
    distance_func = manhattan
elif metric == "Minkowski":
    distance_func = lambda p1, p2: minkowski(p1, p2, minkowski_power)
elif metric == "Cosine":
    distance_func = cosine_distance

# -----------------------------------------------------------------------------
# User Selects a Point (by index) from the Dataset
# -----------------------------------------------------------------------------
st.sidebar.header("Select a Data Point")
selected_index = st.sidebar.slider("Select a point index", 0, len(X)-1, 0,
                                   help="This selects the reference point for which distances to all other points are calculated.")
selected_point = X[selected_index]
st.write(f"### Distances from point {selected_index}: {selected_point}")
st.write("""
The selected point (highlighted in red below) is used as the reference. We compute its 'distance' to every other point using the chosen distance metric.
""")

# -----------------------------------------------------------------------------
# Compute Distances from the Selected Point to All Other Points
# -----------------------------------------------------------------------------
distances = [distance_func(selected_point, X[i]) for i in range(len(X))]
df_distances = pd.DataFrame({"Index": range(len(X)), "Distance": distances})
df_distances = df_distances.sort_values("Distance")

st.write(f"### Nearest Neighbours using {metric} Distance")
st.write("The table below shows the indices and distances of the points nearest to the selected point. Note: The first row will be the point itself (distance 0), so the next row shows its closest 'neighbour'.")
st.dataframe(df_distances.head(10))

# -----------------------------------------------------------------------------
# Plot: Highlight Selected Point and Its Nearest Neighbour
# -----------------------------------------------------------------------------
fig2, ax2 = plt.subplots()
ax2.scatter(df["x"], df["y"], c=df["label"], cmap="viridis", label="Data Points")
ax2.scatter(selected_point[0], selected_point[1], color="red", s=100, label="Selected Point")
# The closest point is the selected point itself, so we take the second row as the nearest neighbour.
nearest_index = int(df_distances.iloc[1]["Index"])
nearest_point = X[nearest_index]
ax2.scatter(nearest_point[0], nearest_point[1], color="orange", s=100, label="Nearest Neighbour")
ax2.legend()
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title(f"Selected Point and Its Nearest Neighbour ({metric} Distance)")
st.pyplot(fig2)

st.write("""
In the plot above:
- The **red** point is the one you selected.
- The **orange** point is its nearest neighbour according to the chosen distance metric.
- The table and plot help illustrate how the chosen metric measures 'difference' between points.
""")

# -----------------------------------------------------------------------------
# Final Explanations
# -----------------------------------------------------------------------------
st.write("""
**Recap:**  
- *Make Moons* is a synthetic dataset that creates two crescent-shaped clusters.  
- **Hyperparameters**:  
  - **Number of Samples**: How many data points are generated.  
  - **Noise Level**: How much randomness (or 'mess') is added to the data.
- **Distance Metrics**:  
  - **Euclidean**: Straight-line distance.
  - **Manhattan**: Distance along grid-like paths.
  - **Minkowski**: A generalisation that lets you dial between Manhattan and Euclidean.
  - **Cosine**: Measures the angle between points (useful when magnitude isn’t as important as direction).

Experiment with the controls in the sidebar to see how the distances and nearest neighbours change. This interactive demo shows you how different ways of measuring 'difference' can lead to very different interpretations of your data.
""")

