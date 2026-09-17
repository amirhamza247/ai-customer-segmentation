"""Streamlit-app som visar segmenten från 03_trana.py. Läser bara, tränar aldrig."""

import altair as alt
import pandas as pd
import streamlit as st

from prepare import FEATURES
from queries import load_segments

st.set_page_config(page_title="Kundsegmentering", layout="wide")


@st.cache_data
def segments():
    return load_segments()


df = segments()
# Relative importance som i 03_trana.py: klustrets snitt mot alla kunders snitt.
# 0,5 = 50 % över snittet. Räknas för alla segment en gång, sidan 2 slår upp sitt.
relative_importance = df.groupby("segment_name")[FEATURES].mean() / df[FEATURES].mean() - 1

sida = st.sidebar.radio("Sida", ["Översikt", "Segmentprofil", "Kundlista", "Kundsökning"])
st.sidebar.caption(f"{len(df):,} kunder i {df.segment_name.nunique()} segment".replace(",", " "))

if sida == "Översikt":
    st.title("Översikt")
    kunder, omsattning = st.columns(2)
    kunder.metric("Antal kunder", f"{len(df):,}".replace(",", " "))
    omsattning.metric("Total omsättning", f"£{df.monetary.sum():,.0f}".replace(",", " "))

    per_segment = df.groupby("segment_name").agg(
        kunder=("customer_id", "size"),
        omsattning=("monetary", "sum"),
    )
    vanster, hoger = st.columns(2)
    vanster.subheader("Kunder per segment")
    vanster.bar_chart(per_segment.kunder, y_label="Kunder", x_label="")
    hoger.subheader("Omsättning per segment")
    hoger.bar_chart(per_segment.omsattning, y_label="£", x_label="")

    st.subheader("Kunderna: recency mot spenderat")
    # Samma vy som rapport/kluster.png, men interaktiv. Log-skala på y: monetary
    # har skevhet 25,3, linjärt trycks alla utom grossisterna ihop mot botten.
    st.altair_chart(
        alt.Chart(df).mark_circle(size=20, opacity=0.6).encode(
            x=alt.X("recency", title="Dagar sedan senaste köp"),
            y=alt.Y("monetary", scale=alt.Scale(type="log"), title="Spenderat (log)"),
            color=alt.Color("segment_name", title="Segment"),
            tooltip=["customer_id", "segment_name", "recency", "frequency", "monetary", "rfm"],
        ).interactive(),  # zoom och panorering
        width="stretch",
    )

elif sida == "Segmentprofil":
    st.title("Segmentprofil")
    valt = st.selectbox("Segment", sorted(df.segment_name.unique()))
    profil = relative_importance.loc[valt]

    st.subheader("Avvikelse från snittet")
    st.bar_chart(profil, horizontal=True, x_label="Andel över/under alla kunders snitt", y_label="")

    st.subheader("Medelvärden mot populationen")
    st.dataframe(
        pd.DataFrame({
            valt: df[df.segment_name == valt][FEATURES].mean(),
            "Alla kunder": df[FEATURES].mean(),
        }).round(0)
    )

    # abs(): "70 % färre köp" är lika intressant som "70 % fler"
    storsta = profil.abs().nlargest(2).index
    st.info(
        f"{valt} avviker mest på "
        + " och ".join(f"**{kolumn}** ({profil[kolumn]:+.0%})" for kolumn in storsta)
        + "."
    )

elif sida == "Kundlista":
    st.title("Kundlista")
    valda = st.multiselect(
        "Segment", sorted(df.segment_name.unique()), default=sorted(df.segment_name.unique())
    )
    urval = df[df.segment_name.isin(valda)]
    st.write(f"{len(urval)} kunder")
    st.dataframe(urval, hide_index=True)
    st.download_button(
        "Ladda ner som CSV",
        urval.to_csv(index=False).encode("utf-8"),
        "kunder.csv",
        "text/csv",
    )

elif sida == "Kundsökning":
    st.title("Kundsökning")
    customer_id = st.number_input("Customer ID", min_value=0, value=12347, step=1)
    traffar = df[df.customer_id == customer_id]

    if traffar.empty:
        st.warning(f"Kund {customer_id} finns inte i segments.")
    else:
        kund = traffar.iloc[0]
        st.subheader(kund.segment_name)
        st.write(f"RFM-kod: **{kund.rfm}** (recency, frequency och monetary, 1–5 där 5 är bäst)")
        st.dataframe(
            pd.DataFrame({
                "Kunden": kund[FEATURES],
                "Alla kunder (snitt)": df[FEATURES].mean(),
                "Segmentet (snitt)": df[df.segment_name == kund.segment_name][FEATURES].mean(),
            }).round(0)
        )
        st.caption(f"LTV {kund.ltv:,.0f} mot snittet {df.ltv.mean():,.0f}".replace(",", " "))
