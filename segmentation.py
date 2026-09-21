import pandas as pd

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
