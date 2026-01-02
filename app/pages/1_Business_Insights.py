import sys
from pathlib import Path

# Fix Path
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.config import ARTIFACTS_DIR, BASELINE_VERSION

st.set_page_config(layout="wide", page_title="Business Analytics - Zimnat")

# --- CSS LOADING ---
with open("app/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- ADVANCED DATA LOADER ---
@st.cache_data
def load_and_enrich_data():
    df = pd.read_parquet(ARTIFACTS_DIR / BASELINE_VERSION / "train_cleaned.parquet")
    meta_cols = ["ID", "join_date", "sex", "marital_status", "birth_year", "branch_code", "occupation_code", "occupation_category_code", "join_year", "age_raw", "age", "join_year_missing", "age_missing", "age_was_clipped"]
    prod_cols = [c for c in df.columns if c not in meta_cols]
    
    # 1. Base Metrics
    df["basket_size"] = df[prod_cols].sum(axis=1)
    df["is_vip"] = df["basket_size"] >= 4
    
    # 2. Date Parsing
    df["join_dt"] = pd.to_datetime(df["join_date"], dayfirst=True, errors='coerce')
    df["join_month"] = df["join_dt"].dt.month_name()
    df["join_month_num"] = df["join_dt"].dt.month
    
    # 3. Tenure & Sleeping Giants
    current_year = 2020 
    df["tenure"] = current_year - df["join_year"]
    df["is_sleeping_giant"] = (df["tenure"] > 5) & (df["basket_size"] <= 1)
    
    # 4. Life Stage
    df["life_stage"] = pd.cut(df["age"], bins=[0, 25, 35, 45, 55, 65, 100], labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"])
    
    return df, prod_cols

try:
    df_raw, prod_cols = load_and_enrich_data()
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# --- HEADER & HORIZONTAL FILTERS ---
st.title("Strategic Business Analytics")

st.markdown("---")
f1, f2, f3 = st.columns([1, 1, 2])

with f1:
    sel_branch = st.multiselect("Branch", sorted(df_raw["branch_code"].unique()))
with f2:
    sel_sex = st.selectbox("Sex", ["All", "M", "F", "Other"])
with f3:
    min_year = int(df_raw["join_year"].min())
    max_year = int(df_raw["join_year"].max())
    sel_year = st.slider("Join Year Scope", min_year, max_year, (2010, 2020))

st.markdown("---")

# --- FILTERING ---
df = df_raw.copy()
if sel_branch: df = df[df["branch_code"].isin(sel_branch)]
if sel_sex != "All": 
    if sel_sex == "M": df = df[df["sex"].str.lower() == "m"]
    elif sel_sex == "F": df = df[df["sex"].str.lower() == "f"]
df = df[(df["join_year"] >= sel_year[0]) & (df["join_year"] <= sel_year[1])]

if df.empty:
    st.warning("No data found matching these filters.")
    st.stop()

# --- KPI CARDS ---
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

total_clients = len(df)
avg_basket = df["basket_size"].mean()
valuable_client_rate = df["is_vip"].mean()
sleeping_giants_count = df["is_sleeping_giant"].sum()

col_kpi1.metric("Active Portfolio", f"{total_clients:,}", "Clients")
col_kpi2.metric("Avg Basket Size", f"{avg_basket:.2f}", "Target: 5.0")
col_kpi3.metric("VIP Segment", f"{valuable_client_rate:.1%}", "Basket >= 4")
col_kpi4.metric("Sleeping Giants", f"{sleeping_giants_count:,}", "Up-sell Potential")

# --- SECTION 1: OPERATIONAL RHYTHM ---
st.header("1. Operational Rhythm")

seasonal_stats = df.groupby(["join_month_num", "join_month"]).size().reset_index(name="Joins")
seasonal_stats = seasonal_stats.sort_values("join_month_num")

if not seasonal_stats.empty:
    peak_row = seasonal_stats.loc[seasonal_stats['Joins'].idxmax()]
    month_peak = peak_row['join_month']
    count_peak = peak_row['Joins']
    avg_seasonal = seasonal_stats['Joins'].mean()
    lift = (count_peak - avg_seasonal) / avg_seasonal
    
    st.markdown(f"""<div class="insight-box">
    <span class="insight-title">Strategic Observation (Scope: {len(df):,} Clients)</span>
    Peak acquisition occurs in <b>{month_peak}</b> with {count_peak} new policies. 
    This represents a <b>+{lift:.1%} increase</b> over the monthly average. 
    Recommendation: Concentrate marketing spend and agent incentives during Q{int((peak_row['join_month_num']-1)/3)+1} to maximize this natural momentum.
    </div>""", unsafe_allow_html=True)

fig_season = px.bar(
    seasonal_stats, x="join_month", y="Joins",
    text="Joins",
    color="Joins", color_continuous_scale="Blues",
    title="Monthly Acquisition Trend"
)
fig_season.update_traces(textposition='outside')
fig_season.update_layout(height=400, xaxis_title=None)
st.plotly_chart(fig_season, use_container_width=True)

# --- SECTION 2: GENDER PREFERENCES ---
st.header("2. Demographic Gap Analysis")

df["sex_norm"] = df["sex"].str.lower()
gender_stats = df.groupby("sex_norm")[prod_cols].mean().T

if 'm' in gender_stats.columns and 'f' in gender_stats.columns:
    gender_stats["Diff"] = gender_stats['m'] - gender_stats['f']
    gender_stats["AbsDiff"] = gender_stats["Diff"].abs()
    sig_gender = gender_stats[gender_stats["AbsDiff"] > 0.02].sort_values("Diff")
    
    if not sig_gender.empty:
        max_gap_prod = sig_gender['AbsDiff'].idxmax()
        gap_val = sig_gender.loc[max_gap_prod, 'Diff']
        direction = "Male" if gap_val > 0 else "Female"
        
        st.markdown(f"""<div class="insight-box">
        <span class="insight-title">Targeting Opportunity (Scope: {len(df):,} Clients)</span>
        Data reveals a significant divergence for <b>{max_gap_prod}</b>, which shows a <b>{abs(gap_val):.1%} higher penetration</b> among {direction} clients.
        Action: Customize communication channels. For this product, prioritize messaging that resonates with the {direction} segment demographics.
        </div>""", unsafe_allow_html=True)

        fig_gender = go.Figure()
        fig_gender.add_trace(go.Bar(y=sig_gender.index, x=sig_gender['m']*100, name='Men', orientation='h', marker_color='#1E88E5'))
        fig_gender.add_trace(go.Bar(y=sig_gender.index, x=sig_gender['f']*-100, name='Women', orientation='h', marker_color='#E91E63'))
        
        fig_gender.update_layout(
            title="Gender Gap (Men vs Women)", xaxis_title="Preference Gap (%)",
            barmode='overlay', height=500,
            xaxis=dict(tickvals=[-40, -20, 0, 20, 40], ticktext=["40% W", "20% W", "0", "20% M", "40% M"])
        )
        st.plotly_chart(fig_gender, use_container_width=True)
    else:
        st.info("Product preferences are balanced across gender demographics.")
else:
    st.warning("Insufficient specific gender data.")

# --- SECTION 3: LIFECYCLE & STRATEGY ---
c1, c2 = st.columns([1, 1])

with c1:
    st.subheader("Lifecycle Curves")
    top_5 = df[prod_cols].sum().nlargest(5).index.tolist()
    evol = df.groupby("life_stage")[top_5].mean().reset_index().melt(id_vars="life_stage")
    
    if top_5:
        top_prod = top_5[0]
        peak_age_txt = df.groupby("life_stage")[top_prod].mean().idxmax()
        
        st.markdown(f"""<div class="insight-box">
        <span class="insight-title">Lifecycle Alignment (Scope: {len(df):,} Clients)</span>
        Penetration for the leading product (<b>{top_prod}</b>) peaks in the <b>{peak_age_txt}</b> segment.
        Strategy: Focus cross-sell efforts on clients approaching this life stage to capture peak buying propensity.
        </div>""", unsafe_allow_html=True)
    
    fig_line = px.line(
        evol, x="life_stage", y="value", color="variable",
        title="Adoption Curves by Age Group", markers=True
    )
    fig_line.update_layout(yaxis_tickformat=".0%", height=400)
    st.plotly_chart(fig_line, use_container_width=True)

with c2:
    st.subheader("Branch Quadrants")
    b_stat = df.groupby("branch_code").agg(Vol=('ID','count'), Qual=('basket_size','mean')).reset_index()
    b_stat = b_stat[b_stat['Vol']>10]
    
    median_vol = b_stat['Vol'].median()
    median_qual = b_stat['Qual'].median()
    
    stars = len(b_stat[(b_stat['Vol'] >= median_vol) & (b_stat['Qual'] >= median_qual)])
    sleepers = len(b_stat[(b_stat['Vol'] < median_vol) & (b_stat['Qual'] < median_qual)])
    
    st.markdown(f"""<div class="insight-box">
    <span class="insight-title">Network Optimization (Scope: {len(df):,} Clients)</span>
    Analysis identifies <b>{stars} 'Star' Agencies</b> driving both high volume and high quality.
    Conversely, <b>{sleepers} agencies</b> are underperforming on both axes.
    Action: Deploy 'Star' branch managers to mentor underperforming branches.
    </div>""", unsafe_allow_html=True)
    
    fig_quad = px.scatter(
        b_stat, x="Vol", y="Qual", color="Qual", size="Vol",
        text="branch_code", title="Volume (X) vs Quality (Y) Matrix",
        color_continuous_scale="Viridis",
        labels={"Vol": "Volume (Clients)", "Qual": "Quality (Basket Size)"}
    )
    fig_quad.add_vline(x=median_vol, line_dash="dash", line_color="grey")
    fig_quad.add_hline(y=median_qual, line_dash="dash", line_color="grey")
    fig_quad.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_quad, use_container_width=True)
