import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

REQUIRED_COLUMNS = ["Invoice", "Quantity", "InvoiceDate", "Price", "Customer ID"]


def clean_transactions(csv_file):
    """Return cleaned transactions and mutually exclusive removal counts."""
    try:
        # Read as strings so identifiers keep leading zeros and a bad numeric
        # value can be handled during cleaning. Keep blank rows to count them.
        # pandas still recognizes default missing-value tokens such as "NA".
        transactions = pd.read_csv(csv_file, dtype="string", skip_blank_lines=False)
    except pd.errors.EmptyDataError as error:
        # "from error" preserves the original cause for debugging while giving
        # the UI a simpler, consistent ValueError to display.
        raise ValueError(
            "The CSV is empty. Upload a file with headers and rows."
        ) from error
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError(
            "Cannot read the file. Use a UTF-8, comma-separated CSV."
        ) from error

    # A list comprehension collects absent names. Membership in a DataFrame
    # checks column names, not the values inside its rows.
    missing = [column for column in REQUIRED_COLUMNS if column not in transactions]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if transactions.empty:
        raise ValueError("The CSV has headers but no transaction rows.")

    for column in REQUIRED_COLUMNS:
        # .str applies a string operation to the whole column. Whitespace-only
        # cells become empty strings, which must be converted to missing values.
        transactions[column] = transactions[column].str.strip().replace("", pd.NA)
    # "coerce" turns unparseable dates into NaT instead of stopping the upload.
    # An explicit format avoids guessing whether a date is day-first or month-first.
    transactions["InvoiceDate"] = pd.to_datetime(
        transactions["InvoiceDate"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    for column in ["Quantity", "Price"]:
        # Invalid numbers become missing. Infinity parses as a number, so remove
        # it explicitly: a positive infinity would otherwise pass a > 0 check.
        transactions[column] = pd.to_numeric(
            transactions[column], errors="coerce"
        ).replace([float("inf"), float("-inf")], pd.NA)

    # Each rule is a boolean Series (one flag per row). Dictionary insertion
    # order defines which reason wins when a row breaks more than one rule.
    # Use isna(), not == pd.NA, to detect missing values. le(0) means <= 0;
    # fillna(False) and na=False keep missing values out of these other rules.
    rules = {
        "Missing customer ID": transactions["Customer ID"].isna(),
        "Missing invoice": transactions["Invoice"].isna(),
        "Cancellation invoice (starts with C)": transactions["Invoice"]
        .str.upper()
        .str.startswith("C", na=False),
        "Missing or invalid invoice date": transactions["InvoiceDate"].isna(),
        "Missing, invalid, or non-finite quantity": transactions["Quantity"].isna(),
        "Zero or negative quantity": transactions["Quantity"].le(0).fillna(False),
        "Missing, invalid, or non-finite price": transactions["Price"].isna(),
        "Zero or negative price": transactions["Price"].le(0).fillna(False),
    }
    # pandas aligns Series by index labels, so the mask uses the data's index.
    keep = pd.Series(True, index=transactions.index)
    removals = {}
    for reason, invalid in rules.items():
        # & combines flags element by element; Python's "and" cannot do this.
        # Summing booleans counts True values. Only count rows still eligible.
        removals[reason] = int((keep & invalid).sum())
        # ~ inverts the flags; &= updates keep so rejected rows stay rejected.
        keep &= ~invalid

    # .loc selects surviving rows. Reset their index without adding the old index
    # as a column. Do not deduplicate: repeated lines may be real transactions.
    return transactions.loc[keep].reset_index(drop=True), removals


def calculate_rfm(cleaned):
    """Return customer RFM values and the reference date for nonempty cleaned data."""
    if cleaned.empty:
        raise ValueError("RFM requires at least one cleaned transaction.")

    # Use the dataset's dates rather than today so results stay reproducible.
    # normalize() removes the time of day: recency measures calendar days,
    # not completed 24-hour periods (which could give a latest purchase 0 days).
    reference_date = cleaned["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    # assign() returns a new DataFrame, leaving the cleaned preview unchanged.
    transactions = cleaned.assign(LineTotal=cleaned["Quantity"] * cleaned["Price"])

    # Named aggregations use output_name=(source_column, operation).
    # nunique counts purchases, not product lines belonging to the same invoice.
    rfm = transactions.groupby("Customer ID", as_index=False).agg(
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("Invoice", "nunique"),
        Monetary=("LineTotal", "sum"),
    )
    # .dt accesses datetime/timedelta operations for an entire Series.
    rfm["Recency"] = (reference_date - rfm["LastPurchase"].dt.normalize()).dt.days
    return rfm[["Customer ID", "Recency", "Frequency", "Monetary"]], reference_date


def scale_rfm(rfm):
    """Log-transform Frequency/Monetary, then standardize RFM without mutating it."""
    if rfm.empty:
        raise ValueError("Scaling requires at least one customer.")

    # Select features explicitly: customer IDs are labels, not measurements.
    # The index identifies rows but is not passed to StandardScaler or K-Means.
    features = rfm.set_index("Customer ID")[["Recency", "Frequency", "Monetary"]]
    if features.isna().any().any() or features.isin(
        [float("inf"), float("-inf")]
    ).any().any():
        raise ValueError("RFM values must be finite numbers before scaling.")

    if (features[["Frequency", "Monetary"]] < 0).any().any():
        raise ValueError("Frequency and Monetary must be nonnegative before log1p.")
    # log1p(x) means natural log(1 + x). It compresses the long upper tails
    # without dropping customers. Recency is not logged because its skew is milder.
    # assign() creates a new frame, preserving original RFM for display/summaries.
    features = features.assign(
        Frequency=np.log1p(features["Frequency"]),
        Monetary=np.log1p(features["Monetary"]),
    )

    # fit_transform learns each column's mean and standard deviation, then applies
    # (value - mean) / standard deviation. This prevents units alone from making
    # spending dominate distance calculations. It does not remove outliers.
    # Constant columns (including a single-customer dataset) become zeros.
    scaler = StandardScaler()
    values = scaler.fit_transform(features)
    # sklearn returns an array by default; restore labels and customer alignment.
    # For future customers, apply the same log1p step and reuse transform(), rather than
    # fitting a new scale that would be inconsistent with a trained model.
    return pd.DataFrame(values, index=features.index, columns=features.columns)


def choose_k(scaled_rfm):
    """Return a suggested K and silhouette scores; retain no fitted model."""
    features = scaled_rfm[["Recency", "Frequency", "Monetary"]]
    # Silhouette needs at least 2 clusters and fewer clusters than customers.
    # Duplicate profiles cannot form separate clusters, even with different IDs.
    max_k = min(8, len(features) - 1, len(features.drop_duplicates()))
    if max_k < 2:
        raise ValueError(
            "K selection needs at least 3 customers and 2 distinct RFM profiles."
        )

    scores = []
    for k in range(2, max_k + 1):
        # These are temporary candidate models, not the final segmentation model.
        # Multiple starts reduce sensitivity to initial centers; a fixed seed
        # makes comparisons reproducible. Customer IDs never enter the fit.
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(features)
        if len(set(labels)) != k:
            continue
        # Score all customers: random sampling can miss small clusters.
        # Higher silhouette means tighter clusters with better separation.
        score = silhouette_score(features, labels)
        scores.append({"K": k, "Silhouette score": score})

    if not scores:
        raise ValueError("No candidate K produced the requested number of clusters.")
    results = pd.DataFrame(scores)
    # idxmax returns the first maximum, so an exact tie favors the smaller K.
    best_k = int(results.loc[results["Silhouette score"].idxmax(), "K"]) # type: ignore to avoid pylance red error line
    return best_k, results


def cluster_customers(scaled_rfm, k):
    """Return the fitted model and cluster IDs indexed by customer ID."""
    features = scaled_rfm[["Recency", "Frequency", "Monetary"]]
    if not features.index.is_unique:
        raise ValueError("Each customer must have exactly one RFM profile.")
    if not 2 <= k <= len(features.drop_duplicates()):
        raise ValueError("K must be between 2 and the number of distinct RFM profiles.")

    # Match the candidate settings so the final fit reproduces the selected run.
    # Fit on scaled features, while retaining original RFM for interpretation.
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(features)
    if len(set(labels)) != k:
        raise ValueError("The final model could not form the requested number of clusters.")
    # fit_predict returns labels in input row order. Attaching the customer index
    # lets the UI join by ID safely, even if the original RFM rows are reordered.
    assignments = pd.Series(labels, index=features.index, name="Cluster")
    return model, assignments


def summarize_clusters(clustered_rfm):
    """Summarize assigned customers using RFM in its original units."""
    # Each input row represents one customer, so size counts customers rather
    # than transactions. Means give each customer equal weight, regardless of
    # how many purchases they made; they are not averages of transaction rows.
    return clustered_rfm.groupby("Cluster", as_index=False).agg(
        CustomerCount=("Customer ID", "size"),
        AverageRecency=("Recency", "mean"),
        AverageFrequency=("Frequency", "mean"),
        AverageMonetary=("Monetary", "mean"),
    )
