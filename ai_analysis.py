import json

from groq import Groq

from segmentation import REQUIRED_COLUMNS

# Check https://console.groq.com/docs/models if Groq retires this model.
MODEL = "openai/gpt-oss-120b"

# The rules keep the model tied to the numbers, so it does not invent business
# meaning the clusters do not support.
SYSTEM_PROMPT = """You explain customer segmentation results to a business user \
who is not a data scientist.

The customers were clustered with K-Means on RFM features:
- Recency: days since the customer's latest purchase. LOWER is better (more recent).
- Frequency: number of distinct invoices. Higher means more repeat purchases.
- Monetary: total spending in the source data's currency.
Averages are per customer. Description compares each cluster with the average \
customer: "Lower"/"Higher" only says below/above the overall mean, not by how much.

Rules:
- Base every statement on the numbers provided. Quote numbers when useful.
- Compare with the overall averages given; do not calculate your own averages.
- Do not assume a currency symbol; say "in the source currency".
- Do not invent segment names. Refer to clusters as "Cluster <number>".
- If two clusters share a Description, say what separates them.
- Present business actions as possibilities, not facts.
- Use the Pareto principle: keep only the few insights that matter most, \
led by which clusters bring most of the revenue (RevenueSharePct) compared with \
their share of customers (CustomerSharePct).
- Write in English, in Markdown, under 120 words. No tables.

Structure:
1. **Key insight:** one sentence on revenue concentration.
2. One bullet per cluster, largest RevenueSharePct first: \
"**Cluster <number>** (<CustomerSharePct>% of customers, <RevenueSharePct>% of \
revenue): <what defines it, in a few words> → <one possible action>."
3. **Focus:** one sentence on where effort likely pays off most."""


def explain_clusters(summary, rfm, k, silhouette, api_key):
    """Return a plain-language Markdown explanation of the cluster summary."""
    # Only aggregated statistics are sent; no customer IDs or transactions.
    # Language models often miscalculate, so compute comparison values here.
    # Count × average spend gives each cluster's total revenue.
    customers = summary["CustomerCount"]
    revenue = customers * summary["AverageMonetary"]
    shares = summary.assign(
        CustomerSharePct=100 * customers / customers.sum(),
        RevenueSharePct=100 * revenue / revenue.sum(),
    )
    table = shares.round(1).to_csv(index=False)
    # Same customer-level means that name_clusters uses for the Description.
    overall = rfm[["Recency", "Frequency", "Monetary"]].mean().round(2)
    user_message = (
        f"K = {k} clusters. Silhouette score = {silhouette:.3f} "
        "(range -1 to 1, higher means better separated clusters).\n"
        f"Total customers: {len(rfm)}. Overall averages: "
        f"Recency {overall['Recency']} days, Frequency {overall['Frequency']} "
        f"invoices, Monetary {overall['Monetary']}.\n\n"
        f"Cluster summary (CSV):\n{table}"
    )
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        # Low temperature gives more consistent, number-focused answers.
        temperature=0.2,
    )
    return response.choices[0].message.content


MAPPING_PROMPT = """You match the columns of an uploaded CSV to the columns a \
customer segmentation pipeline needs. Reply with JSON only.

Required columns:
- "Invoice": order, invoice, or transaction ID. Several rows may share one.
- "Quantity": number of units bought on the row.
- "InvoiceDate": when the purchase happened.
- "Price": price per unit.
- "Customer ID": the customer identifier.

Rules:
- Use exact column names from the CSV, each at most once. Use null if none fits.
- "date_format" is a Python strptime format that parses the example \
InvoiceDate values exactly, for example "%m/%d/%Y %H:%M".

Reply format:
{"columns": {"Invoice": ..., "Quantity": ..., "InvoiceDate": ..., \
"Price": ..., "Customer ID": ...}, "date_format": ...}"""


def suggest_column_mapping(transactions, api_key):
    """Return {required column: CSV column} and the date format the AI suggests."""
    # The header and three example rows are enough to recognize each column,
    # so the rest of the dataset is never sent.
    examples = transactions.dropna(how="all").head(3).to_csv(index=False)
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": MAPPING_PROMPT},
            {"role": "user", "content": f"CSV header and first rows:\n{examples}"},
        ],
        # JSON mode guarantees a reply that json.loads can parse.
        response_format={"type": "json_object"},
        # Zero temperature: the same file should get the same mapping.
        temperature=0,
    )
    try:
        reply = json.loads(response.choices[0].message.content)
        columns, date_format = reply["columns"], reply["date_format"]
        # Keep only matches to real CSV columns. A required column the AI
        # could not match stays absent, so validation names it as missing.
        mapping = {
            target: source
            for target, source in columns.items()
            if target in REQUIRED_COLUMNS
            and isinstance(source, str)
            and source in transactions.columns
        }
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as error:
        raise ValueError("The AI returned an unreadable column mapping.") from error
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("The AI matched one CSV column to several required columns.")
    if not isinstance(date_format, str):
        date_format = None
    return mapping, date_format
