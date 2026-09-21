import streamlit as st

from segmentation import REQUIRED_COLUMNS, calculate_rfm, clean_transactions, scale_rfm


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
        "Each feature is standardized as (value - mean) / standard deviation. "
        "Zero means average; positive values are above average and negative values "
        "are below average. Values are not limited to 0–1. Nonconstant columns "
        "have mean 0 and population standard deviation 1; constant columns become "
        "zeros. Scaling does not remove outliers. Customer IDs are labels only."
    )
    # Restore the ID as a visible column only for display. scaled_rfm itself
    # contains just the three numeric features needed for later clustering.
    st.dataframe(scaled_rfm.reset_index(), hide_index=True)


# This guard runs the UI when executed, but not when another module imports it.
if __name__ == "__main__":
    main()
