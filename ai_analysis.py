from groq import Groq

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
