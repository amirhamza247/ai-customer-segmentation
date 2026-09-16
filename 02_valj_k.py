"""Hjälper oss välja antal kluster k. Tränar ingen slutlig modell."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from databearbetning.prepare import prepare
from databearbetning.queries import load_customers

SNAPSHOT = "2011-12-10"

X = StandardScaler().fit_transform(prepare(load_customers(SNAPSHOT)))

rader = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    rader.append({
        "k": k,
        "inertia": kmeans.inertia_,  # lägre är bättre, men sjunker alltid med k: leta knäet
        "silhouette": silhouette_score(X, kmeans.labels_),  # högre är bättre, -1..1
        "davies_bouldin": davies_bouldin_score(X, kmeans.labels_),  # lägre är bättre
        "minsta_kluster": pd.Series(kmeans.labels_).value_counts().min(),
    })
matt = pd.DataFrame(rader).set_index("k")

print(matt.round(3).to_string())
print(f"\nBäst silhouette:     k={matt.silhouette.idxmax()}")
print(f"Bäst Davies-Bouldin: k={matt.davies_bouldin.idxmin()}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(matt.index, matt.inertia, marker="o")
ax1.set(title="Elbow", xlabel="k", ylabel="Inertia")
ax2.plot(matt.index, matt.silhouette, marker="o")
ax2.set(title="Silhouette", xlabel="k", ylabel="Silhouette-värde")
fig.tight_layout()

Path("rapport").mkdir(exist_ok=True)
fig.savefig("rapport/valj_k.png", dpi=120)
print("\nFigur sparad: rapport/valj_k.png")
