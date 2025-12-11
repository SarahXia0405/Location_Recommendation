# app.py
# ---------------------------------------------------------
# Milk Tea & Dessert Location Recommendation – Streamlit App
# ---------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# ----------------------
# Config & File Paths
# ----------------------
DATA_DIR = "data"  # folder where CSVs live

MASTER_PATH = f"{DATA_DIR}/final_master_clean.csv"
YELP_PATH = f"{DATA_DIR}/yelp_with_comm.csv"
LP_RECO_PATH = f"{DATA_DIR}/lincoln_park_recommendations.csv"  # optional


st.set_page_config(
    page_title="Milk Tea Location Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data(show_spinner=False)
def cached_read_csv(path):
    return pd.read_csv(path)


# ----------------------
# Helper utilities
# ----------------------
def load_csv(path, expected=True):
    try:
        return cached_read_csv(path)
    except FileNotFoundError:
        if expected:
            st.error(f"Could not find `{path}` in the current folder.")
        return None



def pick_col(df, candidates):
    """
    Return the first existing column in df from a list of candidates.
    Raises KeyError if none exist.
    """
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns found: {candidates}")


def ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ----------------------
# Load data
# ----------------------
df_master = load_csv(MASTER_PATH)
df_yelp = load_csv(YELP_PATH)
df_lp = load_csv(LP_RECO_PATH, expected=False)

if df_master is None or df_yelp is None:
    st.stop()

# Normalize column names a bit (strip spaces)
df_master.columns = [c.strip().replace(" ", "_") for c in df_master.columns]
df_yelp.columns = [c.strip().replace(" ", "_") for c in df_yelp.columns]
if df_lp is not None:
    df_lp.columns = [c.strip().replace(" ", "_") for c in df_lp.columns]

# Key columns (with flexible names)
COMM_NAME_COL = pick_col(df_master, ["community_name", "Community_Name"])
COMM_ID_COL = pick_col(df_master, ["community_id", "Community_ID"])

TOTAL_POP_COL = pick_col(df_master, ["total_pop", "Total_Pop"])
PCT_18_34_Z_COL = pick_col(df_master, ["pct_18_34_z", "Pct_18_34_Z"])
PCT_ASIAN_Z_COL = pick_col(df_master, ["pct_asian_z", "Pct_Asian_Z"])
HI_INC_SHARE_Z_COL = pick_col(df_master, ["high_income_share_z", "High_Income_Share_Z"])
CTA_RIDES_Z_COL = pick_col(df_master, ["total_cta_rides_z", "Total_Cta_Rides_Z"])
DIVVY_Z_COL = pick_col(df_master, ["num_divvy_stations_z", "Num_Divvy_Stations_Z"])
CRIME_Z_COL = pick_col(df_master, ["crime_count_z", "Crime_Count_Z"])

FINAL_SCORE_COL = pick_col(df_master, ["final_score_adj", "Final_Score_Adj"])

# Optional competition columns
NUM_COMP_COL = next((c for c in ["num_competitors", "Num_Competitors"] if c in df_master.columns), None)
WHITE_SPACE_COL = next((c for c in ["final_score_white_space", "Final_Score_White_Space"] if c in df_master.columns), None)

# Ensure numeric
df_master = ensure_numeric(
    df_master,
    [
        TOTAL_POP_COL,
        PCT_18_34_Z_COL,
        PCT_ASIAN_Z_COL,
        HI_INC_SHARE_Z_COL,
        CTA_RIDES_Z_COL,
        DIVVY_Z_COL,
        CRIME_Z_COL,
        FINAL_SCORE_COL,
        NUM_COMP_COL,
        WHITE_SPACE_COL,
    ],
)

# Yelp columns
YELP_LAT_COL = pick_col(df_yelp, ["lat", "Latitude"])
YELP_LON_COL = pick_col(df_yelp, ["lon", "Longitude"])
YELP_NAME_COL = pick_col(df_yelp, ["name", "Name"])
YELP_RATING_COL = pick_col(df_yelp, ["rating", "Rating"])
YELP_REVIEWS_COL = pick_col(df_yelp, ["review_count", "Review_Count"])
YELP_COMM_NAME_COL = pick_col(df_yelp, ["community_name", "Community_Name"])


# ----------------------
# Sidebar Navigation
# ----------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Community Rankings",
        "Accessibility",
        "Competitive Landscape",
        "Lincoln Park Recommendations",
        "Radar Chart Explorer",
    ],
)

# Filter for top candidate communities (optional)
top_n = st.sidebar.slider(
    "Highlight top N communities by final score", min_value=3, max_value=20, value=10
)

top_communities = (
    df_master.sort_values(FINAL_SCORE_COL, ascending=False)
    .head(top_n)[COMM_NAME_COL]
    .tolist()
)

# ----------------------
# PAGE 1 – HOME
# ----------------------
if page == "Home":
    st.title("Milk Tea & Dessert Location Intelligence – Chicago")

    st.markdown(
        """
This interactive app summarizes your full analysis pipeline:

1. **Market potential** – demographics, target age (18–34), Asian population, income.
2. **Accessibility** – CTA ridership and Divvy station density.
3. **Safety** – lower relative crime vs other communities.
4. **Competition** – existing bubble-tea & dessert shops from Yelp.
5. **Final recommendation** – best **community + street segments** for a new store.

Use the left sidebar to explore each section like a mini web application.
"""
    )

    # Simple summary KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Communities evaluated", df_master[COMM_ID_COL].nunique())
    with c2:
        st.metric("Competitor shops (Yelp)", len(df_yelp))
    with c3:
        st.metric("Top candidate communities", top_n)
    with c4:
        lincoln_score = float(
            df_master.loc[df_master[COMM_NAME_COL] == "LINCOLN PARK", FINAL_SCORE_COL].iloc[0]
        ) if "LINCOLN PARK" in df_master[COMM_NAME_COL].values else np.nan
        st.metric("Lincoln Park final score", f"{lincoln_score:0.2f}" if not np.isnan(lincoln_score) else "N/A")

# ----------------------
# PAGE 2 – COMMUNITY RANKINGS
# ----------------------
elif page == "Community Rankings":
    st.title("Community Evaluation & Ranking")

    # Ranking barplot
    ranking_df = df_master.sort_values(FINAL_SCORE_COL, ascending=False)

    fig = px.bar(
        ranking_df,
        x=FINAL_SCORE_COL,
        y=COMM_NAME_COL,
        orientation="h",
        color=COMM_NAME_COL,
        color_discrete_sequence=px.colors.sequential.Viridis,
        title="Final Opportunity Score by Community",
    )
    fig.update_layout(showlegend=False, height=800)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Target Population vs Asian Population")
    fig2 = px.scatter(
        df_master,
        x=PCT_18_34_Z_COL,
        y=PCT_ASIAN_Z_COL,
        size=TOTAL_POP_COL,
        color=COMM_NAME_COL,
        hover_name=COMM_NAME_COL,
        title="Z-scores: 18–34 Population vs Asian Population",
    )
    # Highlight top N
    fig2.update_traces(
        selector=lambda t: t.name in top_communities,
        marker=dict(line=dict(width=2, color="black")),
    )
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

# ----------------------
# PAGE 3 – ACCESSIBILITY
# ----------------------
elif page == "Accessibility":
    st.title("Accessibility – CTA & Divvy")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("CTA Rides vs Divvy Stations")
        fig = px.scatter(
            df_master,
            x=CTA_RIDES_Z_COL,
            y=DIVVY_Z_COL,
            size=TOTAL_POP_COL,
            color=COMM_NAME_COL,
            hover_name=COMM_NAME_COL,
            labels={
                CTA_RIDES_Z_COL: "CTA Rides (Z)",
                DIVVY_Z_COL: "Divvy Stations (Z)",
            },
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Top Communities by CTA Rides")
        top_cta = df_master.sort_values(CTA_RIDES_Z_COL, ascending=False).head(15)
        fig_bar = px.bar(
            top_cta,
            x=CTA_RIDES_Z_COL,
            y=COMM_NAME_COL,
            orientation="h",
            color=COMM_NAME_COL,
            labels={CTA_RIDES_Z_COL: "CTA Rides (Z)"},
        )
        fig_bar.update_layout(showlegend=False, height=600)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown(
        """
**Interpretation**: Communities in the upper-right quadrant combine **high transit access**
and **dense micromobility (Divvy)** – excellent for a foot-traffic-driven drink shop.
"""
    )

# ----------------------
# PAGE 4 – COMPETITIVE LANDSCAPE
# ----------------------
elif page == "Competitive Landscape":
    st.title("Competitive Landscape – Bubble Tea & Dessert Shops")

    # Aggregate competition by community
    comp_by_comm = (
        df_yelp.groupby(YELP_COMM_NAME_COL)
        .agg(
            num_shops=(YELP_NAME_COL, "count"),
            avg_rating=(YELP_RATING_COL, "mean"),
            avg_reviews=(YELP_REVIEWS_COL, "mean"),
        )
        .reset_index()
    )

    st.subheader("Competition Heatmap by Community")
    fig = px.treemap(
        comp_by_comm,
        path=[YELP_COMM_NAME_COL],
        values="num_shops",
        color="avg_rating",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=3.8,
        hover_data={"avg_reviews": True},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Competitors Map (All Communities)")
    m = folium.Map(location=[41.88, -87.63], zoom_start=11, tiles="cartodbpositron")

    # Color scale based on rating
    def rating_color(r):
        if pd.isna(r):
            return "gray"
        if r >= 4.5:
            return "green"
        if r >= 4.0:
            return "orange"
        return "red"

    for _, row in df_yelp.iterrows():
        lat = row[YELP_LAT_COL]
        lon = row[YELP_LON_COL]
        if pd.isna(lat) or pd.isna(lon):
            continue

        popup = f"{row[YELP_NAME_COL]}<br>Rating: {row[YELP_RATING_COL]} ({row[YELP_REVIEWS_COL]} reviews)"
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color=rating_color(row[YELP_RATING_COL]),
            fill=True,
            fill_opacity=0.8,
            popup=popup,
        ).add_to(m)

    st_folium(m, width=1000, height=600)

    # Heatmap of competitor density
    st.subheader("Competitor Density Heatmap")
    hm_map = folium.Map(location=[41.88, -87.63], zoom_start=11, tiles="cartodbpositron")
    heat_data = df_yelp[[YELP_LAT_COL, YELP_LON_COL]].dropna().values.tolist()
    if heat_data:
        HeatMap(heat_data, radius=15, blur=10).add_to(hm_map)
    st_folium(hm_map, width=1000, height=600)

# ----------------------
# PAGE 5 – LINCOLN PARK RECOMMENDATIONS
# ----------------------
elif page == "Lincoln Park Recommendations":
    st.title("Recommended Location – Lincoln Park Focus")

    if df_lp is None:
        st.warning("`lincoln_park_recommendations.csv` not found. "
                   "Add it to the folder to see street-level recommendations.")
    lp_mask = df_yelp[YELP_COMM_NAME_COL] == "LINCOLN PARK"
    df_lp_yelp = df_yelp[lp_mask].copy()

    st.subheader("Existing Competitors in Lincoln Park")
    st.dataframe(
        df_lp_yelp[[YELP_NAME_COL, YELP_RATING_COL, YELP_REVIEWS_COL]].sort_values(
            YELP_RATING_COL, ascending=False
        ),
        use_container_width=True,
    )

    st.subheader("Lincoln Park – Competitors & Recommended Spots")

    m_lp = folium.Map(location=[41.92, -87.65], zoom_start=14, tiles="cartodbpositron")

    # Existing shops
    for _, row in df_lp_yelp.iterrows():
        folium.CircleMarker(
            location=[row[YELP_LAT_COL], row[YELP_LON_COL]],
            radius=5,
            color="red",
            fill=True,
            fill_opacity=0.8,
            popup=f"{row[YELP_NAME_COL]} (Rating {row[YELP_RATING_COL]})",
        ).add_to(m_lp)

    # Recommended points (if file exists)
    if df_lp is not None:
        lat_col = pick_col(df_lp, ["lat", "Latitude"])
        lon_col = pick_col(df_lp, ["lon", "Longitude"])
        rank_col = next((c for c in ["rank", "priority_rank", "Priority_Rank"] if c in df_lp.columns), None)
        label_col = next((c for c in ["label", "location_name", "Location_Name"] if c in df_lp.columns), None)

        for _, row in df_lp.iterrows():
            popup = ""
            if label_col and not pd.isna(row[label_col]):
                popup += f"{row[label_col]}<br>"
            if rank_col and not pd.isna(row[rank_col]):
                popup += f"Priority Rank: {int(row[rank_col])}"
            folium.Marker(
                location=[row[lat_col], row[lon_col]],
                popup=popup or "Recommended Spot",
                icon=folium.Icon(color="orange", icon="star"),
            ).add_to(m_lp)

    st_folium(m_lp, width=1000, height=600)

# ----------------------
# PAGE 6 – RADAR CHART EXPLORER
# ----------------------
elif page == "Radar Chart Explorer":
    st.title("Final Opportunity Radar – Community Comparison")

    metrics = {
        "Young Population (18–34, Z)": PCT_18_34_Z_COL,
        "Asian Population (Z)": PCT_ASIAN_Z_COL,
        "High Income Share (Z)": HI_INC_SHARE_Z_COL,
        "CTA Rides (Z)": CTA_RIDES_Z_COL,
        "Divvy Stations (Z)": DIVVY_Z_COL,
        "Crime Count (Z – lower is better)": CRIME_Z_COL,
    }

    # Select community
    communities = df_master[COMM_NAME_COL].sort_values().unique().tolist()
    default_comm = "LINCOLN PARK" if "LINCOLN PARK" in communities else communities[0]
    comm_choice = st.selectbox("Select Community", communities, index=communities.index(default_comm))

    # Build radar data
    row = df_master[df_master[COMM_NAME_COL] == comm_choice].iloc[0]
    values = [row[mcol] for mcol in metrics.values()]
    labels = list(metrics.keys())

    # Shift crime so that higher is better (multiply by -1)
    crime_index = labels.index("Crime Count (Z – lower is better)")
    values[crime_index] = -values[crime_index]

    # Close the loop
    values += values[:1]
    labels_closed = labels + [labels[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=labels_closed,
            fill="toself",
            name=comm_choice,
        )
    )

    # Optionally overlay Lincoln Park for comparison
    if comm_choice != default_comm and default_comm in communities:
        lp_row = df_master[df_master[COMM_NAME_COL] == default_comm].iloc[0]
        lp_vals = [lp_row[mcol] for mcol in metrics.values()]
        lp_vals[crime_index] = -lp_vals[crime_index]
        lp_vals += lp_vals[:1]
        fig.add_trace(
            go.Scatterpolar(
                r=lp_vals,
                theta=labels_closed,
                fill="toself",
                name=default_comm,
                opacity=0.5,
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        height=600,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
**How to read this radar chart:**

- Points farther from the center indicate **better opportunity**  
  (more target customers, higher income, more transit & bikes, *lower* crime).
- The **red area** that covers more surface implies a stronger overall case
  for opening a store in that community.
"""
    )


