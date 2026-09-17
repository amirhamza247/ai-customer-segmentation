"""Visar de fem segmenten från 03_trana.py på recency mot monetary. Tränar ingenting."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from queries import load_segments

df = load_segments()

fig, ax = plt.subplots(figsize=(9, 6))
sns.scatterplot(data=df, x="recency", y="monetary", hue="segment_name",
                s=10, alpha=0.5, linewidth=0, legend=False, ax=ax)

# Median, inte medel: monetary är så skev att medlet hamnar långt från punktmolnet
for segment, (x, y) in df.groupby("segment_name")[["recency", "monetary"]].median().iterrows():
    ax.scatter(x, y, marker="X", s=200, c="black", edgecolors="white")
    ax.annotate(segment, (x, y), xytext=(6, 6), textcoords="offset points", fontweight="bold")

ax.set(yscale="log", title="Segment: recency mot monetary (X = segmentets median)",
       xlabel="Dagar sedan senaste köp", ylabel="Spenderat (log)")
fig.tight_layout()
Path("rapport").mkdir(exist_ok=True)
fig.savefig("rapport/kluster.png", dpi=120)
print("Figur sparad: rapport/kluster.png")
