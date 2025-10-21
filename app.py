# app.py — Bangkok District Rent Explorer
# ───────────────────────────────────────────────────────────────
# Files expected:
#   data/bangkok_rent_listings.csv
#       columns: beds,baths,size_m2,rent_thb,subdistrict,district,province
#   data/bangkok_districts_4326.geojson
#       must contain a district-name column (English or Thai).
#       This script auto-detects common name fields and normalises spellings.

import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import json
from pathlib import Path
import streamlit.components.v1 as components

# 1) Page setup
st.set_page_config("Bangkok Rent Map", layout="wide", page_icon=":house:")

# (Optional) Plausible analytics — change domain or delete
components.html(
    """
    <script defer data-domain="bangkok-rent-app-66.streamlit.app" src="https://plausible.io/js/script.js"></script>
    """,
    height=0,
)

# Compact tables in left column
st.markdown(
    """
    <style>
    .block-container .element-container:has(.dataframe)        {width: fit-content;}
    .block-container .element-container:has(.dataframe) > div  {width: fit-content;}
    .block-container .element-container:has(.dataframe)        {margin: 0 auto;}
    </style>
    """,
    unsafe_allow_html=True
)

# 2) Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_CSV  = DATA_DIR / "bangkok_rent_listings.csv"
GEOJSON  = DATA_DIR / "bangkok_districts_4326.geojson"

# 3) Load data
df_raw = pd.read_csv(RAW_CSV)
gdf_base = gpd.read_file(GEOJSON)

# Ensure numeric types
for c in ("beds", "baths"):
    if c in df_raw.columns:
        df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce").astype("Int64")
df_raw["size_m2"]  = pd.to_numeric(df_raw.get("size_m2"), errors="coerce")
df_raw["rent_thb"] = pd.to_numeric(df_raw.get("rent_thb"), errors="coerce")

# Derived metric
if "price_per_m2" not in df_raw.columns:
    df_raw["price_per_m2"] = df_raw["rent_thb"] / df_raw["size_m2"]

# Clean text
for c in ("district", "subdistrict", "province"):
    if c in df_raw.columns:
        df_raw[c] = df_raw[c].astype(str).str.strip()

# 4) Name normalisation (handles Watthana/Vadhana etc.)
def normalise_bkk_name(s: str) -> str:
    if not isinstance(s, str):
        return s
    t = s.strip().lower()
    t = (
        t.replace("  ", " ")
         .replace("-", " ")
         .replace(" amphoe", "")
         .replace(" khet", "")
         .replace(" khlong toey", " khlong toei")  # unify Toey→Toei
    )
    fixes = {
        "vadhana": "watthana",
        "wadthana": "watthana",
        "watana": "watthana",
        "khlong toi": "khlong toei",
        "khlong toey": "khlong toei",
        "bangrak": "bang rak",
        "pathumwan": "pathum wan",
        "samphanthawongse": "samphanthawong",
    }
    return fixes.get(t, t)

df_raw["district_norm"] = df_raw["district"].astype(str).map(normalise_bkk_name)

# Detect district name column in GeoJSON (English or Thai)
geo_name_cands = [
    # English
    "KHET_EN", "khet_en", "DISTRICT", "district", "NAME", "name",
    # Thai
    "KHET_TH", "khet_th", "NAME_TH", "name_th"
]
geo_name_col = next((c for c in geo_name_cands if c in gdf_base.columns), None)
if geo_name_col is None:
    st.error("Can't find a district name column in the Bangkok GeoJSON. "
             "Expected one of: " + ", ".join(geo_name_cands))
    st.stop()

# Normalised join key on geo side
gdf_base["district_norm"] = gdf_base[geo_name_col].astype(str).map(normalise_bkk_name)

# 5) Sidebar filters
with st.sidebar:
    st.header("Filters")

    bed_opts  = sorted(df_raw["beds"].dropna().unique().tolist()) if "beds" in df_raw else []
    bath_opts = sorted(df_raw["baths"].dropna().unique().tolist()) if "baths" in df_raw else []
    sub_opts  = sorted(df_raw["subdistrict"].dropna().unique().tolist())

    sel_beds  = st.multiselect("Beds", bed_opts, bed_opts) if bed_opts else []
    sel_baths = st.multiselect("Baths", bath_opts, bath_opts) if bath_opts else []
    sel_subs  = st.multiselect("Subdistrict", sub_opts, sub_opts)

    size_min, size_max = float(df_raw["size_m2"].min()), float(df_raw["size_m2"].max())
    rent_min, rent_max = float(df_raw["rent_thb"].min()), float(df_raw["rent_thb"].max())

    size_rng = st.slider(
        "Size (m²)",
        min_value=0.0,
        max_value=max(size_max, 200.0),
        value=(max(0.0, size_min), min(size_max, max(size_max, 200.0))),
        step=1.0,
    )
    rent_rng = st.slider(
        "Rent (THB/mo)",
        min_value=0.0,
        max_value=max(rent_max, 120000.0),
        value=(max(0.0, rent_min), min(rent_max, max(rent_max, 120000.0))),
        step=1000.0,
    )

    metric_labels = {
        "Median Rent": "Median_Rent",
        "Median Rent per m²": "Median_Rent_per_m2",
    }
    metric_label = st.radio("Colour metric", list(metric_labels.keys()))
    metric = metric_labels[metric_label]

# 6) Filter listings
mask = pd.Series(True, index=df_raw.index)
if sel_beds:
    mask &= df_raw["beds"].isin(sel_beds)
if sel_baths:
    mask &= df_raw["baths"].isin(sel_baths)
if sel_subs:
    mask &= df_raw["subdistrict"].isin(sel_subs)

mask &= df_raw["size_m2"].between(size_rng[0], size_rng[1], inclusive="both")
mask &= df_raw["rent_thb"].between(rent_rng[0], rent_rng[1], inclusive="both")

df_f = df_raw[mask].copy()
if df_f.empty:
    st.error("No listings match the current filters.")
    st.stop()

# 7) Aggregate per district (filtered view)
agg = (
    df_f.groupby("district_norm")
        .agg(
            Median_Rent=("rent_thb", "median"),
            Mean_Rent=("rent_thb", "mean"),
            P25_Rent=("rent_thb", lambda s: s.quantile(.25)),
            P75_Rent=("rent_thb", lambda s: s.quantile(.75)),
            Median_Rent_per_m2=("price_per_m2", "median"),
            Listings=("rent_thb", "size"),
        )
        .round(0)
        .reset_index()
)

# 8) Merge stats into geo layer used for plotting
gdf = gdf_base.merge(agg, on="district_norm", how="left")

# 9) Top-10 table
display_metric_col = "Median Rent" if metric == "Median_Rent" else "Median Rent per m²"
top10_table = (
    agg.rename(columns={
        "district_norm": "District",
        "Median_Rent": "Median Rent",
        "Median_Rent_per_m2": "Median Rent per m²",
    })
    .sort_values(display_metric_col, ascending=False)
    .loc[:, ["District", display_metric_col]]
    .head(10)
    .reset_index(drop=True)
    .style
    .hide(axis="index")
    .format({
        "Median Rent": "{:,.0f}",
        "Median Rent per m²": "{:,.0f}",
    })
)

# 10) Plotly choropleth
# Ensure the join key is present in the GeoJSON properties
geojson_obj = json.loads(gdf.to_json())

hover_cols = {
    "District": gdf["district_norm"],
    "Median Rent": ":,.0f THB",
    "Mean Rent": ":,.0f THB",
    "25th Percentile": ":,.0f THB",
    "75th Percentile": ":,.0f THB",
    "Listings": True,
    "district_norm": False,  # hide raw key
}

fig = px.choropleth_mapbox(
    gdf,
    geojson=geojson_obj,
    locations="district_norm",                 # column in gdf
    featureidkey="properties.district_norm",   # property inside geojson features
    color=display_metric_col,
    hover_name="district_norm",
    hover_data=hover_cols,
    color_continuous_scale="YlOrRd",
    mapbox_style="carto-positron",
    center={"lat": 13.7563, "lon": 100.5018},
    zoom=10.3,
    opacity=0.85,
)
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=750)

# 11) Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.title("Bangkok District Rent Explorer")
    st.markdown(
        """
**How this works**

* Filter **beds**, **baths**, optional **subdistricts**, and clamp **size/rent**.
* The map colors each district by your chosen metric (default: **Median Rent per m²**).
* Hover a district for quick stats.

**Glossary**

| Term | Meaning |
|------|---------|
| **Median Rent** | Middle monthly rent of all filtered listings. |
| **Mean Rent** | Average rent across listings. |
| **25th / 75th Percentile** | Cheaper and pricier ends of the market. |
| **Median Rent per m²** | Median monthly rent divided by interior area. |

---
💡 **Moving to Bangkok?**  
No-BS district breakdowns, commute times, rent bands, and pitfalls:  
👉 _Add your CTA/link here (Gumroad/YouTube/etc.)_
        """,
        unsafe_allow_html=True
    )
    st.subheader("Top 10 (current view)")
    st.write(top10_table)
    st.markdown("---")

with col2:
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.markdown(
        """
📺 **[Made by](https://www.youtube.com/@malcolmtalks)**  
☕️ **[Support](https://buymeacoffee.com/malcolmlegy)**
        """,
        unsafe_allow_html=True
    )
