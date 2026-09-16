"""Laddar data/online_retail_II.csv till tabellen transactions i data/kunder.db.

Ingen rensning här: tabellen ska spegla källfilen.
"""

import pandas as pd
from sqlalchemy import create_engine, text

CSV_PATH = "data/online_retail_II.csv"
DB_URL = "sqlite:///data/kunder.db"

df = pd.read_csv(
    CSV_PATH,
    # Invoice och StockCode innehåller bokstäver (C489449, 79323P), läs som text
    dtype={"Invoice": str, "StockCode": str, "Description": str, "Country": str},
)

df = df.rename(
    columns={
        "Invoice": "invoice_no",
        "StockCode": "stock_code",
        "Description": "description",
        "Quantity": "quantity",
        "InvoiceDate": "invoice_date",
        "Price": "unit_price",
        "Customer ID": "customer_id",
        "Country": "country",
    }
)

# Läses som float (13085.0) eftersom kolumnen har NaN; Int64 tillåter NULL
df["customer_id"] = df["customer_id"].astype("Int64")
# SQLite saknar datumtyp; julianday() kräver exakt detta format
df["invoice_date"] = pd.to_datetime(df["invoice_date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
df["quantity"] = df["quantity"].astype("int64")
df["unit_price"] = df["unit_price"].astype("float64")

engine = create_engine(DB_URL)
df.to_sql("transactions", engine, if_exists="replace", index=False, chunksize=10000)

with engine.begin() as conn:
    conn.execute(text("CREATE INDEX idx_transactions_customer_id ON transactions (customer_id)"))
    conn.execute(text("CREATE INDEX idx_transactions_invoice_date ON transactions (invoice_date)"))
    antal = conn.execute(text("SELECT COUNT(*) FROM transactions")).scalar()

print(f"CSV-rader: {len(df):,}  Rader i transactions: {antal:,}")
assert antal == len(df), "Antal rader i databasen matchar inte CSV-filen"
