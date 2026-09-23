from math import ceil, floor, log10

import streamlit as st
from groq import GroqError

from ai_analysis import MODEL as AI_MODEL
from ai_analysis import explain_clusters
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
    st.set_page_config(page_title="Customer Segmentation", layout="wide")
    st.html("""
        <style>
        .stMainBlockContainer {max-width: 1080px; padding-top: 3rem; padding-bottom: 4rem;}
        h1, h2, h3 {letter-spacing: -0.035em; font-weight: 600;}
        h1 {font-size: clamp(2rem, 4vw, 2.8rem); padding-bottom: .4rem;}
        h3 {font-size: 1.3rem; padding-top: 1.5rem;}
        [data-testid="stCaptionContainer"] {opacity: .8;}
        [data-testid="stSidebar"] {border-right: 0;}
        [data-testid="stSidebarContent"] {padding-top: 2rem;}
        [data-testid="stFileUploaderDropzone"] {border-radius: 12px;}
        [data-testid="stButton"] button {border-radius: 10px;}
        [data-testid="stExpander"] details {border: 0; border-radius: 10px;}
        </style>
    """)
    st.title("Customer Segmentation")
    st.caption("Turn customer purchases into clear, useful groups.")
    uploaded_file = st.file_uploader("Upload your CSV", type="csv")
    with st.expander("CSV format and sample dataset"):
        st.caption("Use a UTF-8, comma-separated CSV.")
        st.caption("Required columns: " + ", ".join(REQUIRED_COLUMNS))
        st.caption("InvoiceDate format: YYYY-MM-DD HH:MM:SS")
        st.caption(
            "Dataset: [Online Retail II on Kaggle]"
            "(https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)"
        )
    # Streamlit reruns this function on interaction. Before an upload, the widget
    # returns None; afterward it returns a file-like object pandas can read.
    if uploaded_file is None:
        st.session_state.pop("prepared_upload", None)
        st.session_state.pop("cluster_results", None)
        return

    # Reuse this upload's results when navigating or changing display controls.
    prepared = st.session_state.get("prepared_upload")
    if prepared is None or prepared[0] != uploaded_file.file_id:
        try:
            with st.spinner("Validating and cleaning transactions..."):
                # Tuple unpacking assigns the function's two results in return order.
                cleaned, removals = clean_transactions(uploaded_file)
        except ValueError as error:
            # Show expected validation failures to the user. Catching every Exception
            # here could hide programming errors that we need to investigate.
            st.error(str(error))
            return

        if cleaned.empty:
            st.warning("No valid transactions remain after cleaning.")
            return

        # Customer RFM
        with st.spinner("Calculating customer RFM..."):
            rfm, reference_date = calculate_rfm(cleaned)
        try:
            scaled_rfm = scale_rfm(rfm)
        except ValueError as error:
            st.error(str(error))
            return
        try:
            with st.spinner("Comparing candidate K values..."):
                best_k, scores = choose_k(scaled_rfm)
        except ValueError as error:
            st.warning(str(error))
            return
        prepared = (
            uploaded_file.file_id, cleaned, removals, rfm, reference_date,
            scaled_rfm, best_k, scores,
        )
        st.session_state["prepared_upload"] = prepared
        st.session_state["cluster_results"] = {}
    _, cleaned, removals, rfm, reference_date, scaled_rfm, best_k, scores = prepared
    st.caption(f"Suggested K: {best_k} ? Best silhouette score for this dataset.")

    # User selectable k-value
    # Only offer K values that successfully formed clusters during evaluation.
    # The suggestion is advisory; the widget's returned value controls the fit.
    available_k = [k for k in (2, 3, 4) if k in scores["K"].values]
    if not available_k:
        st.warning("No valid K between 2 and 4 is available for these customers.")
        return
    default_k = best_k if best_k in available_k else available_k[0]
    selected_k = st.selectbox(
        "Number of groups (K)",
        width=240,
        options=available_k,
        index=available_k.index(default_k),
        help="Choose 2–4 clusters. Unavailable values are omitted for this dataset. "
        "Your selection controls customer assignments and the cluster summary.",
    )

    # Keep the existing fit and summary for each K in this upload only.
    cluster_results = st.session_state["cluster_results"]
    if selected_k not in cluster_results:
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
        cluster_results[selected_k] = (model, clustered_rfm, summary)
    model, clustered_rfm, summary = cluster_results[selected_k]
    page = st.sidebar.radio(
        "Page", ["Overview", "Clusters", "Data"], label_visibility="collapsed"
    )

    if page == "Clusters":
        # Cluster summary
        st.subheader("Clusters")
        st.write(f"Selected K: {selected_k} groups. Suggested K: {best_k}.")
        st.caption(
            "Compare group sizes and average days since purchase, purchase counts, "
            "and spending. Very large purchases can affect averages."
        )
        st.caption("Descriptions compare each group with the average customer in this upload.")
        # Format only the display; preserve full precision in the summary DataFrame.
        st.dataframe(
            summary,
            hide_index=True,
            column_config={
                "CustomerCount": st.column_config.NumberColumn(
                    "Customer count", format="%d"
                ),
                "AverageRecency": st.column_config.NumberColumn(
                    "Days since purchase", format="%.2f"
                ),
                "AverageFrequency": st.column_config.NumberColumn(
                    "Purchases", format="%.2f"
                ),
                "AverageMonetary": st.column_config.NumberColumn(
                    "Spending", format="%.2f"
                ),
            },
        )

        with st.expander("How K was suggested"):
            st.caption(
                "Compare K = 2 through 8, limited by the number of customers and distinct "
                "RFM profiles. Temporary K-Means models are fitted for evaluation only. "
                "Silhouette scores range from -1 to 1; higher is better. The highest "
                "score suggests a K, but does not guarantee useful business segments. "
                "Scores use all customers, so large uploads may take longer."
            )
            st.dataframe(scores, hide_index=True)

    elif page == "Data":
        st.subheader("Data")
        st.caption("Explore the records behind your results.")
        with st.expander("Cleaning results"):
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
                {
                    "Removal reason": list(removals),
                    "Rows removed": list(removals.values()),
                }
            )

        with st.expander("Cleaned transactions"):
            st.caption(f"Showing the first {min(len(cleaned), 1000):,} cleaned rows.")
            # Limit only the displayed preview; head() does not change cleaned or counts.
            st.dataframe(cleaned.head(1000), hide_index=True)

        with st.expander("Customer RFM"):
            st.write(
                f"{len(rfm):,} customers. Reference date: {reference_date:%Y-%m-%d}."
            )
            st.caption(
                "Recency: calendar days since the latest purchase (lower means more recent). "
                "Frequency: distinct invoices. Monetary: total Quantity × Price in the "
                "source data's currency. The reference date is one day after the latest "
                "cleaned transaction."
            )
            st.dataframe(rfm, hide_index=True)

        with st.expander("Customer cluster assignments"):
            st.write(
                f"Assigned {len(clustered_rfm):,} customers to {model.n_clusters} clusters."  # type: ignore pylance
            )
            st.caption(
                "The model uses scaled RFM. This table shows original RFM values for "
                "interpretation. Cluster IDs start at 0 and are arbitrary labels, not "
                "rankings."
            )
            st.dataframe(clustered_rfm, hide_index=True)

        with st.expander("Scaled RFM (technical details)"):
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

    elif page == "Overview":
        st.subheader("Overview")
        st.caption("Purchase recency and spending. Each dot is a customer; colors show groups.")
        # Text labels give each cluster a discrete color instead of a numeric gradient.
        chart_data = clustered_rfm.assign(
            Cluster="Cluster " + clustered_rfm["Cluster"].astype(str)
        )
        plot_view = st.radio(
            "Customers shown in plot",
            options=["Typical customers (99%)", "All customers"],
            horizontal=True,
            label_visibility="collapsed",
        )
        # Filter only the plot copy, using original customer-level Monetary values.
        monetary_limit = rfm["Monetary"].quantile(0.99)
        if plot_view == "Typical customers (99%)":
            chart_data = chart_data.loc[chart_data["Monetary"] <= monetary_limit]
            st.caption(
                f"Spending up to {monetary_limit:,.2f} ? 99th percentile"
            )
        hidden_count = len(clustered_rfm) - len(chart_data)
        st.caption(
            f"Showing {len(chart_data):,} of {len(clustered_rfm):,} customers; "
            f"{hidden_count:,} customers hidden from the plot. "
            "Results still include everyone."
        )
        linear_dataset_name = f"linear_customers_k_{selected_k}"
        st.vega_lite_chart(
            spec={
                "data": {"name": linear_dataset_name},
                "datasets": {linear_dataset_name: chart_data},
                "mark": {"type": "circle", "size": 30, "opacity": 0.9},
                "encoding": {
                    "x": {
                        "field": "Recency",
                        "type": "quantitative",
                        "title": "Recency (days)",
                    },
                    "y": {
                        "field": "Monetary",
                        "type": "quantitative",
                        "title": "Monetary (source currency)",
                    },
                    "color": {"field": "Cluster", "type": "nominal"},
                    "tooltip": [
                        {"field": "Customer ID", "type": "nominal"},
                        {"field": "Recency", "type": "quantitative"},
                        {"field": "Monetary", "type": "quantitative", "format": ",.2f"},
                        {"field": "Cluster", "type": "nominal"},
                    ],
                },
                "params": [
                    {"name": "linear_zoom", "select": "interval", "bind": "scales"}
                ],
            },
            width="stretch",
        )

        # Customer segments: logarithmic Monetary axis
        st.subheader("Spending on a log scale")
        st.caption(
            "Same customers. Each step up means 10? more spending."
        )
        # Format axis exponents as superscripts; the Monetary data stays unchanged.
        exponent_label = "format(log(datum.value) / log(10), '.0f')"
        for digit, superscript in zip("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"):
            exponent_label = f"replace({exponent_label}, /{digit}/g, '{superscript}')"
        monetary_ticks = [
            10**power
            for power in range(
                floor(log10(chart_data["Monetary"].min())),
                ceil(log10(chart_data["Monetary"].max())) + 1,
            )
        ]
        # Changing the named dataset also refreshes the chart spec when K changes.
        log_dataset_name = f"customers_k_{selected_k}"
        st.vega_lite_chart(
            spec={
                "data": {"name": log_dataset_name},
                "datasets": {log_dataset_name: chart_data},
                "mark": {"type": "circle", "size": 30, "opacity": 0.9},
                "encoding": {
                    "x": {
                        "field": "Recency",
                        "type": "quantitative",
                        "title": "Recency (days)",
                    },
                    "y": {
                        "field": "Monetary",
                        "type": "quantitative",
                        "title": "Monetary (source currency, log scale)",
                        "scale": {"type": "log", "base": 10},
                        "axis": {
                            "values": monetary_ticks,
                            "labelExpr": "'10' + " + exponent_label,
                        },
                    },
                    "color": {"field": "Cluster", "type": "nominal"},
                    "tooltip": [
                        {"field": "Customer ID", "type": "nominal"},
                        {"field": "Recency", "type": "quantitative"},
                        {"field": "Monetary", "type": "quantitative", "format": ",.2f"},
                        {"field": "Cluster", "type": "nominal"},
                    ],
                },
                "params": [
                    {"name": "log_zoom", "select": "interval", "bind": "scales"}
                ],
            },
            width="stretch",
        )

        # AI explanation
        st.subheader("AI Summary")
        st.caption(
            f"Uses group totals only ? Groq ({AI_MODEL}). No customer records are sent. "
            "Check AI insights against your results."
        )
        try:
            api_key = st.secrets.get("GROQ_API_KEY", "")
        except FileNotFoundError:
            api_key = ""
        if not api_key:
            st.info(
                "To enable AI explanations, run "
                "`cp .streamlit/secrets.example.toml .streamlit/secrets.toml` "
                "and paste your Groq API key into the new file."
            )
            return

        # Streamlit reruns the script on every interaction, so the explanation is
        # kept in session_state. The key ties it to this upload and K, so a stale
        # explanation is never shown after the user changes either.
        analysis_key = (uploaded_file.file_id, selected_k)
        if st.button("Explain clusters with AI", icon=":material/auto_awesome:"):
            silhouette = scores.loc[scores["K"] == selected_k, "Silhouette score"].iloc[
                0
            ]
            try:
                with st.spinner("Asking the AI model..."):
                    explanation = explain_clusters(
                        summary, rfm, selected_k, silhouette, api_key
                    )
            except GroqError as error:
                # Covers a wrong key, rate limits on the free tier, and network errors.
                st.error(f"The AI request failed: {error}")
                return
            st.session_state["ai_explanation"] = (analysis_key, explanation)

        saved = st.session_state.get("ai_explanation")
        if saved is not None and saved[0] == analysis_key:
            st.markdown(saved[1])


# This guard runs the UI when executed, but not when another module imports it.
if __name__ == "__main__":
    main()
