import streamlit as st
import pandas as pd
from pathlib import Path
from neuralprophet import NeuralProphet

# =====================================================
# PAGE SETUP
# =====================================================
st.set_page_config(page_title="FSS Forecasting Tool", layout="wide")

st.markdown(
    "<h1 style='font-size: 64px;'>FSS Forecasting Tool</h1>",
    unsafe_allow_html=True
)

LOGO_PATH = Path("FSSLogo.png")


# =====================================================
# UTILITY FUNCTIONS
# =====================================================
def handle_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names and safely deduplicate them.
    """
    df.columns = (
        df.columns.str.strip()
        .str.replace('\xa0', ' ')
        .str.replace(r'[^a-zA-Z0-9]', ' ', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
        .str.replace(' ', '_')
    )

    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cnt = 1
        for idx in cols[cols == dup].index:
            cols[idx] = f"{dup}_{cnt}"
            cnt += 1

    df.columns = cols
    return df


@st.cache_data
def load_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load and validate uploaded data.
    """
    df = handle_duplicate_columns(df)

    required_columns = {
        "Organization_Code",
        "Account_Number",
        "Fiscal_Year",
        "Current_Month_Actuals",
    }

    missing = required_columns - set(df.columns)
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        st.stop()

    return df


@st.cache_data
def run_fit_predict(input_df: pd.DataFrame):
    """
    Fit NeuralProphet and return model + forecast.
    """
    model = NeuralProphet()
    model.fit(input_df)

    future_df = model.make_future_dataframe(
        input_df,
        n_historic_predictions=True,
        periods=62
    )

    forecast = model.predict(future_df)
    return model, forecast


# =====================================================
# MAIN APP
# =====================================================
def main():

    # --- Section containers (visual layout control)
    st.divider()
    metrics_container = st.container()
    st.divider()
    plot_container = st.container()
    st.divider()
    table_container = st.container()
    st.divider()

    # -------------------------------------------------
    # FILE UPLOAD
    # -------------------------------------------------
    upload_res = st.file_uploader("Upload Fiscal CSV File")

    if upload_res is None:
        return

    df = load_data(pd.read_csv(upload_res))

    # -------------------------------------------------
    # SIDEBAR (LOGO + FILTERS)
    # -------------------------------------------------
    with st.sidebar:
        if LOGO_PATH.exists():
            st.image(LOGO_PATH, use_container_width=True)

        st.header("Data Filters")

        filter_columns = [
            "Organization_Code",
            "Account_Number",
            "Sub_Account_Number",
            "Period_Number",
            "Fiscal_Year",
            "Category_Description",
            "Consolidation_Object_Name",
        ]

        selected_filters = {}
        for col in filter_columns:
            if col not in df.columns:
                continue

            options = sorted(df[col].dropna().unique(), key=str)
            selected_filters[col] = st.multiselect(
                f"Select {col.replace('_', ' ')}",
                options=options,
                key=f"filter_{col}",
            )

        forecast_change = st.number_input(
            "Adjust forecast (%)",
            format="%.2f",
            step=0.25
        )

    # -------------------------------------------------
    # APPLY FILTERS
    # -------------------------------------------------
    filtered_df = df.copy()
    for col, values in selected_filters.items():
        if values:
            filtered_df = filtered_df[filtered_df[col].isin(values)]

    # -------------------------------------------------
    # DATA CLEANING
    # -------------------------------------------------
    filtered_df = filtered_df[~filtered_df["Period_Number"].isin(["BB", "CB"])]

    filtered_df["Period_Number"] = (
        pd.to_numeric(filtered_df["Period_Number"], errors="coerce")
        .replace(13, 12)
        .astype("Int64")
    )

    filtered_df["ds"] = pd.to_datetime(
        filtered_df["Fiscal_Year"].astype(str)
        + "-"
        + filtered_df["Period_Number"].astype(str).str.zfill(2)
        + "-01",
        errors="coerce",
    )

    filtered_df = filtered_df.dropna(subset=["ds"])
    max_dt = filtered_df["ds"].max()

    # -------------------------------------------------
    # FORECAST PREP
    # -------------------------------------------------
    sum_actuals = (
        filtered_df
        .groupby("ds", as_index=False)
        .agg({"Current_Month_Actuals": "sum"})
    )
    sum_actuals["y"] = sum_actuals["Current_Month_Actuals"]
    sum_actuals.drop(columns="Current_Month_Actuals", inplace=True)

    model, predict_df = run_fit_predict(sum_actuals)
    predict_df["yhat1"] *= (1 + forecast_change / 100)

    # -------------------------------------------------
    # PLOT
    # -------------------------------------------------
    with plot_container:
        st.subheader("Actuals & Forecast")

        fig = model.plot(predict_df)
        fig.update_layout(title="")
        fig.update_yaxes(title_text="Amount ($)")
        fig.update_xaxes(title_text="Date")

        st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------
    # YEARLY TOTALS (HOT TAKES)
    # -------------------------------------------------
    predict_df = predict_df[predict_df["ds"] > sum_actuals["ds"].max()]

    actuals_by_year = (
        filtered_df
        .groupby(filtered_df["ds"].dt.year)
        .agg(actuals=("Current_Month_Actuals", "sum"))
    )

    forecast_by_year = (
        predict_df
        .groupby(predict_df["ds"].dt.year)
        .agg(forecast=("yhat1", "sum"))
    )

    yearly_totals = actuals_by_year.join(forecast_by_year, how="outer").fillna(0)
    yearly_totals["total"] = yearly_totals["actuals"] + yearly_totals["forecast"]
    yearly_totals["delta"] = yearly_totals["total"].diff()

    # -------------------------------------------------
    # HOT TAKES
    # -------------------------------------------------
    with metrics_container:
        st.subheader(
            "🔥 Forecast Hot Takes"
            if forecast_change == 0
            else f"🔥 Forecast Hot Takes ({forecast_change:+.2f}%)"
        )

        years = list(yearly_totals.index)[:6]
        tiles_per_row = 3

        for i in range(0, len(years), tiles_per_row):
            cols = st.columns(tiles_per_row)
            for col, year in zip(cols, years[i:i + tiles_per_row]):
                col.metric(
                    label=f"FY {year}",
                    value=f"${yearly_totals.loc[year, 'total']:,.0f}",
                    delta=(
                        None
                        if pd.isna(yearly_totals.loc[year, "delta"])
                        else f"{yearly_totals.loc[year, 'delta']:,.0f}"
                    ),
                    border=True,
                )

    # -------------------------------------------------
    # TABLE + DOWNLOAD
    # -------------------------------------------------
    with table_container:
        st.subheader("Actuals & Forecast by Period and Fiscal Year")

        CMA = (
            filtered_df
            .groupby(["Fiscal_Year", "Period_Number"], as_index=False)
            .agg({"Current_Month_Actuals": "sum"})
            .pivot(index="Period_Number", columns="Fiscal_Year", values="Current_Month_Actuals")
        )

        forecast_pivot = (
            predict_df
            .assign(month=predict_df["ds"].dt.month, year=predict_df["ds"].dt.year)
            .pivot(index="month", columns="year", values="yhat1")
        )

        df_combined = pd.concat(
            [CMA.add_prefix("Actual_"), forecast_pivot.add_prefix("Forecast_")],
            axis=1,
        )

        df_combined.index.name = "Period"
        df_combined.loc["Totals"] = df_combined.sum()

        st.dataframe(df_combined.style.format("{:,.0f}"), use_container_width=True)

        st.download_button(
            "Download CSV",
            df_combined.to_csv().encode("utf-8"),
            "forecast.csv",
            "text/csv",
            icon=":material/download:",
        )


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    main()