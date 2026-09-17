"""Sista transformeringen före modellen. Används av både träningen och appen.

StandardScaler ligger medvetet utanför: den lär sig medelvärden från datan
och måste sparas tillsammans med modellen.
    Träning:  X = scaler.fit_transform(prepare(df))
    Inferens: X = scaler.transform(prepare(df))
"""

import numpy as np

# rfm är inte med: den är ett betyg för att läsa resultatet, inte ett mått att klustra på
FEATURES = ["recency", "frequency", "monetary", "ltv"]
# monetary har skevhet 25,3; utan log drar enskilda grossister iväg en centroid
LOG_FEATURES = ["frequency", "monetary", "ltv"]


def prepare(df):
    saknas = [kol for kol in FEATURES if kol not in df.columns]
    if saknas:
        raise ValueError(f"Kolumner saknas i prepare(): {saknas}. Förväntade {FEATURES}")

    df = df[FEATURES].copy()  # ändra aldrig anroparens DataFrame
    df[LOG_FEATURES] = np.log1p(df[LOG_FEATURES])
    # Returnera i fast ordning: scikit-learn matchar kolumner på position, inte namn
    return df[FEATURES]


if __name__ == "__main__":
    from queries import load_customers

    kunder = load_customers()
    X = prepare(kunder)
    print(X.describe().round(2))
    print("Skevhet före:", kunder[FEATURES].skew().round(1).to_dict())
    print("Skevhet efter:", X.skew().round(1).to_dict())

    assert list(X.columns) == FEATURES
    assert "rfm" not in X.columns
    assert kunder.monetary.max() > 600_000, "prepare() får inte ändra indata"
    assert X.recency.equals(kunder.recency)
    try:
        prepare(kunder.drop(columns="ltv"))
        raise AssertionError("ValueError förväntades")
    except ValueError as e:
        print("OK:", e)
