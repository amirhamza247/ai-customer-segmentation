import streamlit as st

from segmentation import (
    REQUIRED_COLUMNS,
    calculate_rfm,
    choose_k,
    clean_transactions,
    cluster_customers,
    name_clusters,
    scale_rfm,
    summarize_clusters,
)


def main():
    st.title("Customer transactions and RFM")
    st.write("Upload a UTF-8, comma-separated CSV to validate and clean transactions.")
    st.caption("Required columns: " + ", ".join(REQUIRED_COLUMNS))
    st.caption("InvoiceDate format: YYYY-MM-DD HH:MM:SS")
    # Streamlit reruns this function on interaction. Before an upload, the widget
    # returns None; afterward it returns a file-like object pandas can read.
    uploaded_file = st.file_uploader("Transaction CSV", type="csv")
    if uploaded_file is None:
        return

    try:
        with st.spinner("Validating and cleaning transactions..."):
            # Tuple unpacking assigns the function's two results in return order.
            cleaned, removals = clean_transactions(uploaded_file)
    except ValueError as error:
        # Show expected validation failures to the user. Catching every Exception
        # here could hide programming errors that we need to investigate.
        st.error(str(error))
        return

    removed = sum(removals.values())
    # Inside an f-string, :, formats a number with thousands separators.
    st.write(
        f"Read {len(cleaned) + removed:,} rows. "
        f"Removed {removed:,} rows. Kept {len(cleaned):,} rows."
    )
    st.caption(
        "Rules run in the order below. Each removed row is counted only under "
        "its first matching reason. Duplicate rows are kept."
    )
    st.table(
        {"Removal reason": list(removals), "Rows removed": list(removals.values())}
    )

    if cleaned.empty:
        st.warning("No valid transactions remain after cleaning.")
        return

    st.subheader("Cleaned transactions")
    st.caption(f"Showing the first {min(len(cleaned), 1000):,} cleaned rows.")
    # Limit only the displayed preview; head() does not change cleaned or counts.
    st.dataframe(cleaned.head(1000), hide_index=True)




# Customer RFM
    with st.spinner("Calculating customer RFM..."):
        rfm, reference_date = calculate_rfm(cleaned)
    st.subheader("Customer RFM")
    st.write(f"{len(rfm):,} customers. Reference date: {reference_date:%Y-%m-%d}.")
    st.caption(
        "Recency: calendar days since the latest purchase (lower means more recent). "
        
        "Frequency: distinct invoices. Monetary: total Quantity × Price in the "
        "source data's currency. The reference date is one day after the latest "
        "cleaned transaction."
    )
    st.dataframe(rfm, hide_index=True)



# Scaled RFM
    st.subheader("Scaled RFM")
    try:
        scaled_rfm = scale_rfm(rfm)
    except ValueError as error:
        st.error(str(error))
        return
    st.caption(
        "Frequency and Monetary first use log1p(x) = ln(1 + x) to compress large "
        "values; Recency is not log-transformed. All three features are then "
        "standardized as (value - mean) / standard deviation. For Frequency and "
        "Monetary, the mean and standard deviation are calculated after log1p. "
        "Zero means average on that scale; positive values are above average and negative values "
        "are below average. Values are not limited to 0–1. Nonconstant columns "
        "have mean 0 and population standard deviation 1; constant columns become "
        "zeros. Scaling does not remove outliers. Customer IDs are labels only."
    )
    # Restore the ID as a visible column only for display. scaled_rfm itself
    # contains just the three numeric features needed for later clustering.
    st.dataframe(scaled_rfm.reset_index(), hide_index=True)

    st.subheader("Suggested number of clusters")
    st.caption(
        "Compare K = 2 through 8, limited by the number of customers and distinct "
        "RFM profiles. Temporary K-Means models are fitted for evaluation only. "
        "Silhouette scores range from -1 to 1; higher is better. The highest "
        "score suggests a K, but does not guarantee useful business segments. "
        "Scores use all customers, so large uploads may take longer."
    )
    try:
        with st.spinner("Comparing candidate K values..."):
            best_k, scores = choose_k(scaled_rfm)
    except ValueError as error:
        st.warning(str(error))
        return
    st.write(f"Suggested K: {best_k} (highest silhouette score among candidates).")
    st.dataframe(scores, hide_index=True)




# User selectable k-value
    # Only offer K values that successfully formed clusters during evaluation.
    # The suggestion is advisory; the widget's returned value controls the fit.
    available_k = [k for k in (2, 3, 4) if k in scores["K"].values]
    if not available_k:
        st.warning("No valid K between 2 and 4 is available for these customers.")
        return
    default_k = best_k if best_k in available_k else available_k[0]
    selected_k = st.selectbox(
        "K for final clustering",
        options=available_k,
        index=available_k.index(default_k),
        help="Choose 2–4 clusters. Unavailable values are omitted for this dataset. "
        "Your selection controls customer assignments and the cluster summary.",
    )
    
    
    
    
# Customer cluster assignments
    st.subheader("Customer cluster assignments")
    try:
        with st.spinner("Training the final K-Means model..."):
            model, assignments = cluster_customers(scaled_rfm, selected_k)
    except ValueError as error:
        st.error(str(error))
        return
    # Join by customer ID, not row position. Validation guards against accidental
    # duplicate IDs multiplying rows. The original RFM table stays unchanged.
    clustered_rfm = rfm.join(assignments, on="Customer ID", validate="one_to_one")
    summary = name_clusters(summarize_clusters(clustered_rfm), rfm)
    clustered_rfm = clustered_rfm.join(
        summary.set_index("Cluster")["Description"],
        on="Cluster",
        validate="many_to_one",
    )
    st.write(
        f"Assigned {len(clustered_rfm):,} customers to {model.n_clusters} clusters." # type: ignore pylance
    )
    st.caption(
        "The model uses scaled RFM. This table shows original RFM values for "
        "interpretation. Cluster IDs start at 0 and are arbitrary labels, not "
        "rankings. Descriptions compare each cluster's original RFM means with "
        "the means across all customers in this upload. Lower Recency means "
        "more recent purchases; exact equality is described as Average. "
        "Descriptions summarize groups, not every member, and may repeat."
    )
    st.dataframe(clustered_rfm, hide_index=True)




# Cluster summary
    st.subheader("Cluster summary")
    st.caption(
        "Customer count and mean RFM per cluster, using original values: "
        "Recency in days, Frequency in invoices, and Monetary in the source "
        "data's currency. Each customer has equal weight. Averages can be "
        "influenced by unusually large values."
    )
    # Format only the display; preserve full precision in the summary DataFrame.
    st.dataframe(
        summary,
        hide_index=True,
        column_config={
            "CustomerCount": st.column_config.NumberColumn("Customer count", format="%d"),
            "AverageRecency": st.column_config.NumberColumn("Average Recency", format="%.2f"),
            "AverageFrequency": st.column_config.NumberColumn("Average Frequency", format="%.2f"),
            "AverageMonetary": st.column_config.NumberColumn("Average Monetary", format="%.2f"),
        },
    )


# This guard runs the UI when executed, but not when another module imports it.
if __name__ == "__main__":
    main()
