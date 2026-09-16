import altair as alt
import streamlit as st
from databearbetning.queries import load_segments

@st.cache_data
def hamta_segment():
return load_segments()

df = hamta_segment()

diagram = alt.Chart(df).mark_circle(size=20, opacity=0.6).encode(
x=alt.X("recency", title="Dagar sedan senaste köp"),
y=alt.Y("monetary", scale=alt.Scale(type="log"), title="Spenderat (log)"),
color=alt.Color("segment_name", title="Segment"),
tooltip=["customer_id", "segment_name", "recency", "frequency", "monetary", "rfm"],
).interactive() # zoom och panorering

st.altair_chart(diagram, use_container_width=True)
