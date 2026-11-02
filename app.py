import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="CovidLens | COVID-19 EDA", page_icon="🦠", layout="wide")
st.title("🦠 CovidLens: COVID-19 Exploratory Data Analysis")

# ---------------- DATA LOADING & CLEANING ----------------
@st.cache_data
def load_data(path="owid-covid-data.csv"):
    cols = ["location", "continent", "date", "new_cases", "new_deaths",
            "total_cases", "total_deaths", "total_vaccinations", "population"]
    df = pd.read_csv(path, usecols=cols, parse_dates=["date"])
    raw_rows = len(df)
    df = df.drop_duplicates()
    df = df.dropna(subset=["continent"])             # remove aggregates (World, Asia...)
    df[["new_cases", "new_deaths"]] = df[["new_cases", "new_deaths"]].fillna(0)
    df = df[(df["new_cases"] >= 0) & (df["new_deaths"] >= 0)]
    df = df.sort_values(["location", "date"])
    df["total_vaccinations"] = df.groupby("location")["total_vaccinations"].ffill()
    # ---------------- FEATURE CALCULATION ----------------
    df["cases_per_million"] = df["total_cases"] / df["population"] * 1_000_000
    df["fatality_rate"] = df["total_deaths"] / df["total_cases"] * 100
    df["cases_7day_avg"] = df.groupby("location")["new_cases"].transform(
        lambda s: s.rolling(7, min_periods=1).mean())
    return df, raw_rows

from pathlib import Path

URL = "https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv"
FILE = Path(__file__).parent / "owid-covid-data.csv"

AGGREGATES = ["World", "Asia", "Europe", "Africa", "North America", "South America",
              "Oceania", "European Union (27)", "European Union", "High-income countries",
              "Low-income countries", "Lower-middle-income countries",
              "Upper-middle-income countries", "International"]

@st.cache_data
def load_data():
    if not FILE.exists():                       # download once if missing
        pd.read_csv(URL).to_csv(FILE, index=False)

    df = pd.read_csv(FILE, parse_dates=["date"])
    raw_rows = len(df)

    if "country" in df.columns:                 # new OWID files use 'country'
        df = df.rename(columns={"country": "location"})

    df = df.drop_duplicates()
    df = df[~df["location"].isin(AGGREGATES)]   # remove World / continents / income groups

    for c in ["new_cases", "new_deaths", "total_cases", "total_deaths",
              "total_vaccinations", "population"]:
        if c not in df.columns:
            df[c] = 0                           # create missing columns so the app won't crash

    df[["new_cases", "new_deaths"]] = df[["new_cases", "new_deaths"]].fillna(0)
    df = df[(df["new_cases"] >= 0) & (df["new_deaths"] >= 0)]
    df = df.sort_values(["location", "date"])
    df["total_vaccinations"] = df.groupby("location")["total_vaccinations"].ffill()
    df["cases_per_million"] = df["total_cases"] / df["population"].replace(0, pd.NA) * 1_000_000
    df["fatality_rate"] = df["total_deaths"] / df["total_cases"].replace(0, pd.NA) * 100
    df["cases_7day_avg"] = df.groupby("location")["new_cases"].transform(
        lambda s: s.rolling(7, min_periods=1).mean())
    return df, raw_rows

df, raw_rows = load_data()

# ---------------- FILTERING ----------------
st.sidebar.header("Filters")
countries = st.sidebar.multiselect("Country", sorted(df["location"].unique()),
                                   default=["India", "United States", "Brazil"])
dates = st.sidebar.date_input("Date range", [df["date"].min(), df["date"].max()])
f = df[df["location"].isin(countries)]
if len(dates) == 2:
    f = f[(f["date"] >= pd.to_datetime(dates[0])) & (f["date"] <= pd.to_datetime(dates[1]))]

# ---------------- KPIs ----------------
latest = f.groupby("location").last()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Cases", f"{int(latest['total_cases'].sum()):,}")
k2.metric("Total Deaths", f"{int(latest['total_deaths'].sum()):,}")
k3.metric("Fatality Rate", f"{latest['total_deaths'].sum() / latest['total_cases'].sum() * 100:.2f}%")
k4.metric("Total Vaccinations", f"{int(latest['total_vaccinations'].sum()):,}")

# ---------------- TREND ANALYSIS ----------------
st.subheader("Daily New Cases (7-day average)")
st.plotly_chart(px.line(f, x="date", y="cases_7day_avg", color="location"),
                use_container_width=True)

monthly = f.groupby([f["date"].dt.to_period("M").astype(str), "location"])["new_cases"].sum().reset_index()
st.subheader("Monthly New Cases")
st.plotly_chart(px.bar(monthly, x="date", y="new_cases", color="location"),
                use_container_width=True)

# ---------------- TOP COUNTRIES ----------------
st.subheader("Top 10 Countries by Total Cases")
top10 = df.groupby("location")["total_cases"].max().nlargest(10).reset_index()
st.plotly_chart(px.bar(top10, x="total_cases", y="location", orientation="h"),
                use_container_width=True)

# ---------------- CORRELATION ----------------
st.subheader("Correlation Heatmap")
corr = f[["new_cases", "new_deaths", "total_vaccinations", "cases_per_million"]].corr()
st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="Blues"),
                use_container_width=True)

# ---------------- DATA QUALITY & EXPORT ----------------
with st.expander("Data Quality Summary"):
    st.write(f"Raw rows: {raw_rows:,} | Clean rows: {len(df):,} | Rows removed: {raw_rows - len(df):,}")
st.download_button("Download filtered data (CSV)", f.to_csv(index=False), "covid_filtered.csv")