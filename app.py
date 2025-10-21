# app.py — Bangkok District Rent Explorer
# ───────────────────────────────────────────────────────────────
# Files expected in ./data:
#   - bangkok_rent_listings.csv
#       columns: beds,baths,size_m2,rent_thb,subdistrict,district,province
#   - bangkok_districts_4326.geojson
#       district (khet) polygons; any name field is fine — auto-detected

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

# ───────────────────────────────────────────────────────────────
# 1) Page setup
st.set_page_config("Bangkok Rent Map", layout="wide", page_icon=":house:")

# (Optional) Plausible analytics — change/remove as you like
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

# ───────────────────────────────────────────────────────────────
# 2) Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_CSV  = DATA_DIR / "bangkok_rent_listings.csv"
GEOJSON  = DATA_DIR / "bangkok_districts_4326.geojson"

# ───────────────────────────────────────────────────────────────
# 3) Loaders (cached)
@st.cache_data
def load_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data
def load_gdf(path: Path) -> gpd.GeoDataFrame:
    return gpd.read_file(path)

try:
    df_raw = load_df(RAW_CSV)
except Exception as e:
    st.error(f"Failed to read CSV at {RAW_CSV}: {e}")
    st.stop()

try:
    gdf_base = load_gdf(GEOJSON)
except Exception as e:
    st.error(f"Failed to read GeoJSON at {GEOJSON}: {e}")
    st.stop()

# Ensure numeric types
for c in ("beds", "baths"):
    if c in df_raw.columns:
        df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce").astype("Int64")
df_raw["size_m2"]  = pd.to_numeric(df_raw.get("size_m2"), errors="coerce")
df_raw["rent_thb"] = pd.to_numeric(df_raw.get("rent_thb"), errors="coerce")

# HARD CAP: drop any listing with baths > 5 (never shown/used anywhere)
if "baths" in df_raw.columns:
    df_raw = df_raw[(df_raw["baths"].isna()) | (df_raw["baths"] <= 5)].copy()

# Derived metric
if "price_per_m2" not in df_raw.columns:
    df_raw["price_per_m2"] = df_raw["rent_thb"] / df_raw["size_m2"]

# Strip text fields
for c in ("district", "subdistrict", "province"):
    if c in df_raw.columns:
        df_raw[c] = df_raw[c].astype(str).str.strip()

# ───────────────────────────────────────────────────────────────
# 4) Normalisation + cleaning

# District normaliser (handles Watthana/Vadhana, Toei/Toey, spacing, etc.)
def normalise_bkk_district(s: str) -> str:
    if not isinstance(s, str):
        return s
    t = s.strip().lower()
    t = (
        t.replace("  ", " ")
         .replace("-", " ")
         .replace(" amphoe", "")
         .replace(" khet", "")
    )
    fixes = {
        "vadhana": "watthana",
        "wadthana": "watthana",
        "watana": "watthana",
        "khlong toey": "khlong toei",
        "khlong toi": "khlong toei",
        "bangrak": "bang rak",
        "pathumwan": "pathum wan",
        "samphanthawongse": "samphanthawong",
        "bang kholaem": "bang kho laem",   # common slip
        "ratburana": "rat burana",
    }
    return fixes.get(t, t)

# Subdistrict cleaner
SUB_DROP_PREFIXES = ("studio ",)  # drop “Studio …”
SUB_FIXES = {
    "saphan song": "saphan sung",       # typo
    # If a pure district name appears in subdistrict col, drop it (unless truly valid)
    "yan nawa": None,
    "bang sue": None,
    "bang na": None,                    # keep only if you know it's the khwaeng, else drop
}
def clean_subdistrict(s: str):
    if not isinstance(s, str):
        return s
    t = s.strip()
    low = t.lower().replace("  ", " ")
    if low.startswith(SUB_DROP_PREFIXES):
        return None
    if low in SUB_FIXES:
        return SUB_FIXES[low]
    return t

df_raw["district_norm"] = df_raw["district"].astype(str).map(normalise_bkk_district)
df_raw["subdistrict"]   = df_raw["subdistrict"].map(clean_subdistrict)
df_raw = df_raw[df_raw["subdistrict"].notna()]  # drop junk rows we nulled

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
gdf_base["district_norm"] = gdf_base[geo_name_col].astype(str).map(normalise_bkk_district)

# Canonical district list (from GeoJSON, so options match the map)
DISTRICT_OPTIONS = sorted(gdf_base["district_norm"].dropna().unique().tolist())

# ───────────────────────────────────────────────────────────────
# 5) Sidebar filters (District above Subdistrict)
with st.sidebar:
    st.header("Filters")

    # District selector (defaults to all in GeoJSON so joins are always valid)
    sel_districts = st.multiselect("District (khet)", DISTRICT_OPTIONS, DISTRICT_OPTIONS)

    # Subdistrict options limited by selected districts from current data
    sub_opts_df = df_raw[df_raw["district_norm"].isin(sel_districts)]
    sub_opts = sorted(sub_opts_df["subdistrict"].dropna().unique().tolist())
    sel_subs = st.multiselect("Subdistrict (khwaeng)", sub_opts, sub_opts)

    bed_opts  = sorted(df_raw["beds"].dropna().unique().tolist()) if "beds" in df_raw else []
    bath_opts = sorted(df_raw["baths"].dropna().unique().tolist()) if "baths" in df_raw else []

    sel_beds  = st.multiselect("Beds", bed_opts, bed_opts) if bed_opts else []
    sel_baths = st.multiselect("Baths", bath_opts, bath_opts) if bath_opts else []

    size_series = df_raw["size_m2"].dropna()
    rent_series = df_raw["rent_thb"].dropna()
    size_min, size_max = float(size_series.min()), float(size_series.max())
    rent_min, rent_max = float(rent_series.min()), float(rent_series.max())

    size_rng = st.slider(
        "Size (m²)", min_value=0.0, max_value=max(size_max, 200.0),
        value=(max(0.0, size_min), min(size_max, max(size_max, 200.0))), step=1.0
    )
    rent_rng = st.slider(
        "Rent (THB/mo)", min_value=0.0, max_value=max(rent_max, 120000.0),
        value=(max(0.0, rent_min), min(rent_max, max(rent_max, 120000.0))), step=1000.0
    )

    metric_labels = {
        "Median Rent": "Median_Rent",
        "Median Rent per m²": "Median_Rent_per_m2",
    }
    metric_label = st.radio("Colour metric", list(metric_labels.keys()))
    metric = metric_labels[metric_label]

    if st.button("Reset filters"):
        st.experimental_rerun()

# ───────────────────────────────────────────────────────────────
# 6) Apply filters
mask = pd.Series(True, index=df_raw.index)
if sel_districts:
    mask &= df_raw["district_norm"].isin(sel_districts)
if sel_subs:
    mask &= df_raw["subdistrict"].isin(sel_subs)
if sel_beds:
    mask &= df_raw["beds"].isin(sel_beds)
if sel_baths:
    mask &= df_raw["baths"].isin(sel_baths)
mask &= df_raw["size_m2"].between(size_rng[0], size_rng[1], inclusive="both")
mask &= df_raw["rent_thb"].between(rent_rng[0], rent_rng[1], inclusive="both")

df_f = df_raw[mask].copy()
if df_f.empty:
    st.error("No listings match the current filters.")
    st.stop()

# ───────────────────────────────────────────────────────────────
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

# Pretty display names for map & tables
agg_disp = agg.rename(columns={
    "Median_Rent": "Median Rent",
    "Mean_Rent": "Mean Rent",
    "P25_Rent": "25th Percentile",
    "P75_Rent": "75th Percentile",
    "Median_Rent_per_m2": "Median Rent per m²",
})

# ───────────────────────────────────────────────────────────────
# 8) Merge stats into geo layer used for plotting
gdf = gdf_base.merge(agg_disp, on="district_norm", how="left")
gdf["District"] = gdf["district_norm"].str.title()

# Warn which districts have no data for current view
display_metric_col = "Median Rent" if metric == "Median_Rent" else "Median Rent per m²"
missing = gdf.loc[gdf[display_metric_col].isna(), ["district_norm"]].drop_duplicates()
if not missing.empty:
    st.warning("No data for: " + ", ".join(missing["district_norm"].str.title().tolist()))

# ───────────────────────────────────────────────────────────────
# 9) Top-10 table (based on current metric)
top10_table = (
    gdf[["District", display_metric_col]]
      .dropna(subset=[display_metric_col])
      .sort_values(display_metric_col, ascending=False)
      .head(10)
      .reset_index(drop=True)
      .style
      .hide(axis="index")
      .format({
          "Median Rent": "{:,.0f}",
          "Median Rent per m²": "{:,.0f}",
      })
)

# ───────────────────────────────────────────────────────────────
# 10) Subdistrict drill-down (table + download)
sub_agg = (
    df_f.groupby(["district_norm", "subdistrict"], dropna=False)
        .agg(
            Median_Rent=("rent_thb", "median"),
            Median_Rent_per_m2=("price_per_m2", "median"),
            Listings=("rent_thb", "size"),
        )
        .reset_index()
)
sub_agg_disp = sub_agg.copy()
sub_agg_disp["District"] = sub_agg_disp["district_norm"].str.title()
sub_agg_disp.rename(columns={
    "subdistrict": "Subdistrict",
    "Median_Rent": "Median Rent",
    "Median_Rent_per_m2": "Median Rent per m²",
}, inplace=True)

metric_for_sub = "Median Rent" if metric == "Median_Rent" else "Median Rent per m²"

# ───────────────────────────────────────────────────────────────
# 11) Plotly choropleth — safe hover_data and guards
if display_metric_col not in gdf.columns:
    st.error(f"Missing metric column in map dataframe: {display_metric_col}")
    st.stop()
if gdf[display_metric_col].notna().sum() == 0:
    st.error("No statistics available for the current filters (all NaN).")
    st.stop()

geojson_obj = json.loads(gdf.to_json())
hover_data = {
    "District": True,
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
    color=display_metric_col,                  # pretty name column
    hover_name="District",
    hover_data=hover_data,
    color_continuous_scale="YlOrRd",
    mapbox_style="carto-positron",
    center={"lat": 13.7563, "lon": 100.5018},
    zoom=10.3,
    opacity=0.85,
)
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=750)
fig.update_coloraxes(colorbar=dict(title=display_metric_col, tickformat=","))

# ───────────────────────────────────────────────────────────────
# 12) Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.title("Bangkok District Rent Explorer")
    st.markdown(
        """
**How this works**

* Filter **District** (khet) and **Subdistrict** (khwaeng), plus **beds/baths** and **size/rent**.
* The map colors each district by your chosen metric (default: **Median Rent per m²**).
* Hover a district for quick stats. Subdistricts are available in the table below.

**Glossary**

| Term | Meaning |
|------|---------|
| **Median Rent** | Middle monthly rent of all filtered listings. |
| **Mean Rent** | Average rent across listings. |
| **25th / 75th Percentile** | Cheaper and pricier ends of the market. |
| **Median Rent per m²** | Median monthly rent divided by interior area. |
        """,
        unsafe_allow_html=True
    )

    st.subheader("Top 10 Districts (current view)")
    st.write(top10_table)

    st.markdown("---")
    st.markdown("### Subdistricts (khwaeng)")
    # Scope selector for drilldown (All or a single district)
    dist_opts = ["All Bangkok"] + sorted(sub_agg_disp["District"].unique().tolist())
    chosen_scope = st.selectbox("Scope", dist_opts, index=0)

    if chosen_scope == "All Bangkok":
        sub_view = (sub_agg_disp
                    .sort_values(metric_for_sub, ascending=False)
                    .loc[:, ["Subdistrict", "District", metric_for_sub, "Listings"]]
                    .head(15))
    else:
        sub_view = (sub_agg_disp[sub_agg_disp["District"] == chosen_scope]
                    .sort_values(metric_for_sub, ascending=False)
                    .loc[:, ["Subdistrict", metric_for_sub, "Listings"]]
                    .head(15))

    st.dataframe(
        sub_view.style.format({
            "Median Rent": "{:,.0f}",
            "Median Rent per m²": "{:,.0f}",
            "Listings": "{:d}",
        }),
        use_container_width=True,
    )

    st.download_button(
        "Download subdistrict table (CSV)",
        data=sub_view.to_csv(index=False).encode("utf-8"),
        file_name="bangkok_subdistricts.csv",
        mime="text/csv"
    )

with col2:
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.markdown(
        """
📺 **[Made by](https://www.youtube.com/@malcolmtalks)**  
💡  **[Moving to Bangkok?](https://malcolmproducts.gumroad.com/l/yjwzkr)**

        """,
        unsafe_allow_html=True
    )