# AI Customer Segmentation

A simple Streamlit application that segments customers from transaction data using **RFM analysis** and **K-Means clustering**.

The project is currently a proof of concept. A user uploads a transaction CSV, the app cleans the data, calculates customer RFM values, prepares the features for machine learning, compares possible cluster counts, and lets the user choose the final number of clusters.

## What the app does

```text
CSV upload
   ↓
Validation and cleaning
   ↓
RFM calculation
   ↓
log1p transformation (Frequency and Monetary)
   ↓
StandardScaler
   ↓
Evaluate K = 2–8 with silhouette score
   ↓
Suggested K
   ↓
User selects final K = 2–4
   ↓
K-Means clustering
   ↓
Customer assignments and cluster summary
```

Original RFM values are preserved for display and interpretation. Transformed and standardized values are used for clustering.

## Requirements

- Git
- Python 3.14 or newer
- uv

Check your installations:

```bash
git --version
python --version
uv --version
```

If you do not have uv installed, see the official uv installation guide.

## Setup

### 1. Clone the repository

```bash
git clone git@github.com:amirhamza247/ai-customer-segmentation.git
cd ai-customer-segmentation
```

### 2. Switch to the development branch

The current application is on the `dev` branch:

```bash
git switch dev
```

### 3. Install dependencies

```bash
uv sync
```

This creates the project environment and installs the locked dependencies.

### 4. Add your Groq API key (optional)

The AI explanation of the clusters uses Groq. Get a free key at https://console.groq.com/keys, then:

```bash
cp .streamlit/secrets.example.toml .streamlit/secrets.toml
```

Paste your key between the quotes in `.streamlit/secrets.toml`. That file is in `.gitignore`, so the key is never committed. Without a key, everything except the AI explanation still works.

### 5. Start the app

```bash
uv run streamlit run main.py
```

Streamlit will print a local address in the terminal, usually `http://localhost:8501`. Open it in your browser.

## CSV format

The app is built for the **Online Retail II** dataset. Download `online_retail_II.csv` from Kaggle:
https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci

CSV files are in `.gitignore`, so keep the file anywhere on your computer and upload it in the app.

Upload a UTF-8, comma-separated CSV containing these required columns:

| Column | Description |
| --- | --- |
| `Invoice` | Invoice/order identifier |
| `Quantity` | Quantity purchased |
| `InvoiceDate` | Transaction date and time |
| `Price` | Price per item |
| `Customer ID` | Customer identifier |

`InvoiceDate` must use:

```text
YYYY-MM-DD HH:MM:SS
```

Example:

```csv
Invoice,Quantity,InvoiceDate,Price,Customer ID
536365,6,2025-12-01 08:26:00,2.55,17850
536366,6,2025-12-01 08:28:00,3.39,17850
536367,8,2025-12-01 08:34:00,2.75,13047
```

Extra columns are allowed.

## Data cleaning

Before segmentation, the app removes rows with missing customer IDs or invoices, cancellation invoices beginning with `C`, invalid dates, invalid/non-finite quantities or prices, and zero or negative quantities or prices.

Duplicate rows are intentionally kept. The app displays how many rows were removed for each reason.

## RFM analysis

Each customer is represented by:

- **Recency** — calendar days since the customer's latest purchase. Lower means more recent.
- **Frequency** — number of distinct invoices.
- **Monetary** — total `Quantity × Price`.

The Recency reference date is one day after the latest cleaned transaction date, making results reproducible instead of dependent on today's date.

## Clustering

Before K-Means, Recency is left untransformed, while Frequency and Monetary use `log1p` to reduce the effect of their long upper tails. All three features are then standardized with `StandardScaler`.

The app evaluates **K=2 through K=8** using silhouette score and shows the highest-scoring K as a suggestion. The user can choose **K=2, K=3, or K=4** for final clustering. Invalid choices are automatically omitted for small datasets.

The selected K controls the final customer assignments and cluster summary.

Cluster IDs such as `0`, `1`, and `2` are arbitrary labels, not rankings.

## Project structure

```text
ai-customer-segmentation/
├── main.py            # Streamlit UI
├── segmentation.py    # Cleaning, RFM, scaling and clustering
├── pyproject.toml     # Project metadata and dependencies
├── uv.lock            # Locked dependencies
├── .python-version    # Python version
├── AGENTS.md          # Instructions for coding agents
└── README.md
```

## Development

Run Ruff:

```bash
uv run ruff check .
```

Apply safe Ruff fixes:

```bash
uv run ruff check . --fix
```

Format with Black:

```bash
uv run black .
```

Before committing:

```bash
uv run ruff check .
uv run black --check .
git diff
```

## Main technologies

Python, Streamlit, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, and uv.

## Current status

The current POC supports CSV upload and validation, transaction cleaning, RFM feature engineering, log transformation and scaling, silhouette-based K comparison, user-selectable K from 2–4, K-Means clustering, and cluster summaries in original RFM units.

Human-readable cluster naming and final cluster visualizations are planned next.
