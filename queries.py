"""SQL-frågor mot databaserna."""

import pandas as pd
from sqlalchemy import create_engine, text

# Hela datan, stannar lokalt (138 MB, över GitHubs gräns på 100 MB)
engine = create_engine("sqlite:///data/kunder.db")
# Bara tabellen segments (under 1 MB), checkas in så att den publicerade appen har data
app_engine = create_engine("sqlite:///data/app.db")
# Dagen efter sista transaktionen i datasetet. Recency räknas härifrån.
SNAPSHOT = "2011-12-10"

# En rad per kund. date() på fakturadatumet ger hela dagar, så en kund som
# handlade dagen före :snapshot får recency 1, inte 0,47.
CUSTOMERS_SQL = """
WITH kunder AS (
    SELECT
        customer_id,
        CAST(julianday(:snapshot) - julianday(date(MAX(invoice_date))) AS INTEGER) AS recency,
        COUNT(DISTINCT invoice_no) AS frequency,
        SUM(quantity * unit_price) AS monetary,
        julianday(:snapshot) - julianday(date(MIN(invoice_date))) AS dagar_som_kund
    FROM transactions
    WHERE customer_id IS NOT NULL
        AND quantity > 0
        AND unit_price > 0
        AND invoice_no NOT LIKE 'C%'
    GROUP BY customer_id
)
SELECT
    customer_id,
    recency,
    frequency,
    monetary,
    -- customer_id som sista sorteringsnyckel: NTILE delar lika värden godtyckligt
    -- (t.ex. alla 1 623 engångskunder), så utan den kan betyget ändras mellan körningar
    NTILE(5) OVER (ORDER BY recency DESC, customer_id)
    || NTILE(5) OVER (ORDER BY frequency, customer_id)
    || NTILE(5) OVER (ORDER BY monetary, customer_id) AS rfm,
    -- Golv på 90 dagar: annars räknas en kund som köpte igår upp till 365 gånger sitt köp per år
    monetary / (MAX(dagar_som_kund, 90) / 365.0) * 3 AS ltv
FROM kunder
ORDER BY customer_id
"""


def load_customers(snapshot=SNAPSHOT):
    """Kunddata per kund. snapshot som 'YYYY-MM-DD'."""
    return pd.read_sql(text(CUSTOMERS_SQL), engine, params={"snapshot": snapshot})


def load_segments():
    """Tabellen segments som 03_trana.py skriver: en rad per kund med kluster och segmentnamn."""
    return pd.read_sql("SELECT * FROM segments", app_engine)


if __name__ == "__main__":
    df = load_customers()
    print(df.head())
    print(df.describe().round(1))
    assert len(df) == 5878, len(df)
    assert df.rfm.str.fullmatch(r"[1-5]{3}").all()
    assert df.recency.min() >= 1
    # Senaste köp ska ge högst R-betyg
    assert df.loc[df.recency.idxmin(), "rfm"][0] == "5"
    assert df.loc[df.monetary.idxmax(), "rfm"][2] == "5"
