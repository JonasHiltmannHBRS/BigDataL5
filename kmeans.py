import numpy as np
import matplotlib.pyplot as plt
import os

# Define dataset
points = {
    'A': (2, 10),
    'B': (2, 5),
    'C': (8, 4),
    'D': (5, 8),
    'E': (7, 5),
    'F': (6, 4),
    'G': (1, 2),
    'H': (4, 9)
}
labels = list(points.keys())
X = np.array(list(points.values()))

# Initial centroids: A (2,10), D (5,8), G (1,2)
initial_centroids = np.array([
    points['A'],
    points['D'],
    points['G']
])

def euclidean_distance(a, b):
    return np.linalg.norm(a - b, axis=1)

def assign_clusters(X, centroids):
    clusters = {i: [] for i in range(len(centroids))}
    for idx, point in enumerate(X):
        distances = euclidean_distance(point, centroids)
        closest = np.argmin(distances)
        clusters[closest].append(idx)
    return clusters

def update_centroids(clusters, X):
    new_centroids = []
    for cluster in clusters.values():
        cluster_points = X[cluster]
        new_centroids.append(np.mean(cluster_points, axis=0))
    return np.array(new_centroids)

def plot_clusters(X, clusters, centroids, step, desc):
    colors = ['deepskyblue', 'navy', 'orange']
    markers = ['s', 'D', 'o']
    
    plt.figure(figsize=(6, 6))
    for i, cluster_indices in clusters.items():
        cluster_points = X[cluster_indices]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    color=colors[i], marker=markers[i], label=f'Cluster {i+1}')
    
    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], color='red',
                s=150, edgecolor='black', marker='o', label='Centroids')
    
    # Label the points
    for idx, label in enumerate(labels):
        plt.text(X[idx, 0] + 0.1, X[idx, 1], label, fontsize=9)
    
    plt.title(f'Step {step}: {desc}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(0, 10)
    plt.ylim(0, 11)
    plt.grid(True)
    plt.legend()
    filename = f'kmeans_step_{step}.png'
    plt.savefig(filename)
    plt.show()

# --- Begin Visualization Process ---

# Step 1: Initial seed points
step = 1
centroids = initial_centroids.copy()
empty_clusters = {i: [] for i in range(len(centroids))}
plot_clusters(X, empty_clusters, centroids, step, 'Initial seed points')

# Step 2: Assign to nearest centroid
step += 1
clusters = assign_clusters(X, centroids)
plot_clusters(X, clusters, centroids, step, 'Assign to nearest seed point')

# Step 3: Update centroids
step += 1
centroids = update_centroids(clusters, X)
plot_clusters(X, clusters, centroids, step, 'Update cluster centroids')

# Step 4: Reassign objects
step += 1
clusters = assign_clusters(X, centroids)
plot_clusters(X, clusters, centroids, step, 'Reassign objects')

# Step 5: Final centroid update
step += 1
centroids = update_centroids(clusters, X)
plot_clusters(X, clusters, centroids, step, 'Final centroids')

print("All steps visualized and saved as images.")
