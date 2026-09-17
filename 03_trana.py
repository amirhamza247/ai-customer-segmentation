"""Tränar den slutliga modellen, namnger klustren och sparar allt.

Det här är det enda skriptet som anropar fit. Appen laddar models/modell.pkl.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from prepare import FEATURES, prepare
from queries import SNAPSHOT, app_engine, load_customers

K = 5  # vald i 02_valj_k.py

# 1. Data
df = load_customers()
scaler = StandardScaler()
X = scaler.fit_transform(prepare(df))

# 2. Modell
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10).fit(X)
df["cluster"] = kmeans.labels_

# 3. Relative importance på otransformerade värden: hur mycket klustrets snitt
#    avviker från alla kunders snitt. 0,5 = 50 % högre, -0,5 = 50 % lägre.
relative_importance = df.groupby("cluster")[FEATURES].mean() / df[FEATURES].mean() - 1


# 4. Namn. Sätts utifrån klustrens värden, inte klusternummer, eftersom
#    KMeans kan numrera om klustren om datan eller scikit-learn ändras.
# ponytail: reglerna förutsätter K=5, skriv om dem om K ändras
def namnge(profil):
    kvar = profil.copy()
    namn = {}
    for segment, kolumn, valj in [
        ("Champions", "monetary", "idxmax"),  # spenderar mest
        ("Lost", "recency", "idxmax"),  # längst sedan senaste köp
        ("At risk", "recency", "idxmax"),  # också länge sedan, men köpte mer än Lost
        ("Loyal", "frequency", "idxmax"),  # av de två aktiva: handlar oftast
        # Aktiva men få köp. Inte "New customers": bara 31 % köpte första gången inom 90 dagar,
        # och modellen har ingen variabel för hur länge någon varit kund.
        ("Occasional", "frequency", "idxmin"),
    ]:
        cluster = getattr(kvar[kolumn], valj)()
        namn[int(cluster)] = segment
        kvar = kvar.drop(cluster)
    return namn


assert K == 5, "namnge() har regler för exakt fem kluster"
segment_names = namnge(df.groupby("cluster")[FEATURES].mean())
df["segment_name"] = df.cluster.map(segment_names)

sammanfattning = df.groupby("segment_name").agg(
    kunder=("customer_id", "size"),
    recency=("recency", "median"),
    frequency=("frequency", "median"),
    monetary=("monetary", "median"),
    ltv=("ltv", "median"),
    vanligaste_rfm=("rfm", lambda s: s.mode()[0]),
)
print("Medianvärden per segment:")
print(sammanfattning.round(0).sort_values("monetary", ascending=False).to_string())

fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(
    relative_importance.rename(index=segment_names),
    annot=True, fmt=".2f", cmap="RdBu_r", center=0,
    vmin=-1, vmax=1,  # Champions ligger på +4,8; utan tak blir alla andra vita
    ax=ax,
)
ax.set(title="Relative importance per segment (avvikelse från snittet)", ylabel="")
fig.tight_layout()
Path("rapport").mkdir(exist_ok=True)
fig.savefig("rapport/relative_importance.png", dpi=120)

# 5. Spara modellen: allt appen behöver för att klassa en ny kund
Path("models").mkdir(exist_ok=True)
joblib.dump(
    {
        "scaler": scaler,
        "kmeans": kmeans,
        "features": FEATURES,
        "segment_names": segment_names,
        "snapshot": SNAPSHOT,
    },
    "models/modell.pkl",
)

# 6. Tabellen appen läser
kolumner = ["customer_id", "recency", "frequency", "monetary", "ltv", "rfm", "cluster", "segment_name"]
df[kolumner].to_sql("segments", app_engine, if_exists="replace", index=False)

print("\nKunder per kluster:")
for cluster, antal in df.cluster.value_counts().sort_index().items():
    varning = "  VARNING: färre än 50 kunder" if antal < 50 else ""
    print(f"  {cluster} {segment_names[cluster]:<14} {antal:>5}{varning}")

print("\nSparat: models/modell.pkl, data/app.db, rapport/relative_importance.png")
