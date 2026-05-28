import streamlit as st
import pandas as pd
from pathlib import Path
from neuralprophet import NeuralProphet

st.set_page_config(page_title="FSS Forecasting Tool", layout="wide")

# Configuration
st.markdown(
    "<h1 style='font-size: 64px;'>FSS Forecasting Tool</h1>",
    unsafe_allow_html=True
)

LOGO_PATH = Path(r"FSSLogo.png")

# ----------------------------
# Utility Functions
# ----------------------------
def handle_duplicate_columns(df):
    """Clean and deduplicate column names"""
    df.columns = (
        df.columns.str.strip()
        .str.replace('\xa0', ' ')
        .str.replace(r'[^a-zA-Z0-9]', ' ', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
        .str.replace(' ', '_')
    )

    cols = pd.Series(df.columns)
    duplicates = cols[cols.duplicated()].unique()
    for dup in duplicates:
        cnt = 1
        for idx in cols[cols == dup].index:
            cols[idx] = f"{dup}_{cnt}"
            cnt += 1
    df.columns = cols
    return df


@st.cache_resource
def load_data(df):
    """Load and preprocess data with error handling"""
    try:
        df = handle_duplicate_columns(df)

        required_columns = {
            'Organization_Code', 'Account_Number',
            'Fiscal_Year', 'Current_Month_Actuals'
        }
        missing = required_columns - set(df.columns)
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
            st.stop()

        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()


@st.cache_resource
def run_fit_predict(input_df, forecast_periods):
    """Fit NeuralProphet and return forecast"""
    m = NeuralProphet(
        growth="linear",
        n_changepoints=0,
        trend_reg=0,
        yearly_seasonality=True,
    )
    metrics = m.fit(input_df, freq="MS")
    df_future = m.make_future_dataframe(
        input_df,
        n_historic_predictions=True,
        periods=forecast_periods
    )

    forecast = m.predict(df_future)
    return [m, forecast]


# ----------------------------
# Main App
# ----------------------------
def main():
    a, b = st.columns(2)
    c, d = st.columns(2)

    st.divider()
    metrics_container = st.container()
    st.divider()
    plot_container = st.container()
    st.divider()
    table_container = st.container()
    st.divider()

    st.write("https://analytics.uaccess.arizona.edu/analytics/saw.dll?Dashboard&PortalPath=%2Fshared%2FFinancial%2F_portal%2FGeneral%20-%20Financial%20Management&page=Transactions")
    upload_res = st.file_uploader("Upload Transactions CSV File")
    if upload_res is not None:
        df = pd.read_csv(upload_res)
        df = load_data(df)

        # Sidebar with logo and filters
        with st.sidebar:
            try:
                if LOGO_PATH.exists():
                    st.image(str(LOGO_PATH), use_container_width=True)
                else:
                    st.warning(f"Logo not found at: {LOGO_PATH}")
            except Exception as e:
                st.error(f"Logo loading error: {str(e)}")

            st.header("Data Filters")

            filter_columns = [
                'Organization_Code',
                'Account_Number',
                'Sub_Account_Number',
                'Period_Number',
                'Fiscal_Year',
                'Category_Description',
                'Consolidation_Object_Name'
            ]

            selected_filters = {}
            for col in filter_columns:
                try:
                    options = df[col].dropna().unique()
                    if col == "Object_Code":
                        options = [str(int(x)).zfill(4) if str(x).isdigit() else str(x) for x in options]

                    try:
                        options = sorted(options, key=lambda x: float(x))
                    except (ValueError, TypeError):
                        options = sorted(options, key=lambda x: str(x))

                    selected = st.multiselect(
                        label=f"Select {col.replace('_', ' ')}",
                        options=options,
                        key=f"filter_{col}"
                    )
                    selected_filters[col] = selected
                except KeyError:
                    pass

            forecast_change = st.number_input("To change the forecast, enter a percentage here. Use a negative number (e.g., -3%) to decrease it or a positive number to increase it:",format="%.2f",step=0.25)

        # Apply filters
        filtered_df = df.copy()
        for col, values in selected_filters.items():
            if values:
                filtered_df = filtered_df[filtered_df[col].isin(values)]

        filtered_df.to_csv("Account Detail_ Transactions.csv", index=False)

        # ----------------------------
        # Data Cleaning for Forecast
        # ----------------------------
        filtered_df = filtered_df[~filtered_df['Period_Number'].isin(['BB', 'CB'])]

        # Convert period numbers to numeric and clean up
        filtered_df['Period_Number'] = pd.to_numeric(filtered_df['Period_Number'], errors='coerce')
        filtered_df['Period_Number'] = filtered_df['Period_Number'].replace(13, 12).astype('Int64')

        # ----------------------------
        # Actuals Table
        # ----------------------------
        periods_sum = (
            filtered_df
            .groupby(["Fiscal_Year", "Period_Number"], as_index=False)
            .agg({"Current_Month_Actuals": "sum"})
        )

        CMA_pivot = periods_sum.pivot(
            index='Period_Number',
            columns='Fiscal_Year',
            values='Current_Month_Actuals'
        )
        CMA_pivot.columns = [str(int(c)) for c in CMA_pivot.columns]

        CMA_pivot.loc["Totals"] = CMA_pivot.sum()
        CMA_pivot_styled = CMA_pivot.style.format(lambda x: f"{x:,.0f}")

        # Build valid YYYY-MM-DD strings
        # Ensure no NaN values remain after conversion
        filtered_df = filtered_df.dropna(subset=['Period_Number', 'Fiscal_Year'])

        # Convert to regular int (not nullable Int64)
        filtered_df['Period_Number'] = filtered_df['Period_Number'].astype(int)
        filtered_df['Fiscal_Year'] = filtered_df['Fiscal_Year'].astype(int)

        # Map fiscal period to calendar month (Period 1 = July, Period 12 = June)
        fiscal_to_calendar_month = {
            1: 7, 2: 8, 3: 9, 4: 10, 5: 11, 6: 12,
            7: 1, 8: 2, 9: 3, 10: 4, 11: 5, 12: 6
        }

        # Periods 7-12 (Jan-Jun) are in the same calendar year as the fiscal year
        # Periods 1-6 (Jul-Dec) are in the previous calendar year
        filtered_df['calendar_month'] = filtered_df['Period_Number'].map(fiscal_to_calendar_month)
        filtered_df['calendar_year'] = filtered_df['Fiscal_Year']
        filtered_df.loc[filtered_df['Period_Number'] <= 6, 'calendar_year'] = filtered_df['Fiscal_Year'] - 1

        # Build date string
        filtered_df['date_string'] = (
                filtered_df['calendar_year'].astype(str)
                + '-'
                + filtered_df['calendar_month'].astype(str).str.zfill(2)
                + '-01'
        )

        # Convert to datetime
        filtered_df['ds'] = pd.to_datetime(filtered_df['date_string'], format='%Y-%m-%d', errors='coerce')

        # Drop invalid rows
        if filtered_df['ds'].isna().any():
            st.warning("⚠️ Some rows had invalid dates and were dropped.")
            filtered_df = filtered_df.dropna(subset=['ds'])

# Identify the maximum date in the dataset

        max_dt = filtered_df['ds'].max()

        # ----------------------------
        # Dynamic Forecast Length
        # ----------------------------

        # Identify most recent fiscal year
        max_fiscal_year = filtered_df['Fiscal_Year'].max()

        # Identify max period within that fiscal year
        max_period = (
            filtered_df[filtered_df['Fiscal_Year'] == max_fiscal_year]
            ['Period_Number']
            .max()
        )

        # Months remaining in that fiscal year
        PERIODS_PER_YEAR = 12
        months_remaining_in_fy = PERIODS_PER_YEAR - max_period

        # Additional forecast years
        FORECAST_YEARS = 5
        additional_months = FORECAST_YEARS * PERIODS_PER_YEAR

        forecast_periods = months_remaining_in_fy + additional_months + 6


        # ----------------------------
        # Prepare Forecast Data
        # ----------------------------
        sum_actuals = filtered_df.groupby('ds', as_index=False).agg({"Current_Month_Actuals": 'sum'})
        sum_actuals['y'] = sum_actuals['Current_Month_Actuals']
        sum_actuals = sum_actuals.drop(columns='Current_Month_Actuals')

        # Ensure only ds and y columns
        sum_actuals = sum_actuals[['ds', 'y']].copy()

        # Ensure correct types
        sum_actuals['ds'] = pd.to_datetime(sum_actuals['ds'])
        sum_actuals['y'] = pd.to_numeric(sum_actuals['y'], errors='coerce')

        # Drop any NaN rows
        sum_actuals = sum_actuals.dropna(subset=['ds', 'y'])

        # Sort by date
        sum_actuals = sum_actuals.sort_values('ds').reset_index(drop=True)

        # Check minimum rows
        if len(sum_actuals) < 10:
            st.error(f"Not enough data points for forecasting ({len(sum_actuals)} rows). Need at least 10.")
            st.stop()

        res = run_fit_predict(sum_actuals, forecast_periods)
        predict_df = res[1]
        m = res[0]
        predict_df['yhat1'] = predict_df["yhat1"] * (1 + (forecast_change/100))

        # ----------------------------
        # Plot Below
        # ----------------------------

        with plot_container:
            st.subheader(
                "Actuals & Forecast"
                if forecast_change == 0
                else f"Actuals & Forecast {forecast_change:+.2f}%"
            )

            new_labels = {"yhat1": "Forecast", "Actual": "Actuals"}
            my_plot = m.plot(predict_df)
            my_plot.update_layout(
                title=dict(
                    text="",
                    x=0.5
                )
            )
            my_plot.for_each_trace(lambda t: t.update(name = new_labels[t.name]))
            my_plot.update_yaxes(title_text="Amount ($)")
            my_plot.update_xaxes(title_text="Date")
            st.plotly_chart(my_plot)

        # Add Year and Month Columns
        predict_df['year'] = predict_df['ds'].dt.year
        predict_df['month'] = predict_df['ds'].dt.month

        # Show only future periods
        max_row = sum_actuals['ds'].max()
        predict_df = predict_df[predict_df.ds > max_row]

        predict_pivot = predict_df.pivot(
            index='month',
            columns='year',
            values='yhat1'
        )
        predict_pivot.columns = [str(int(c)) for c in predict_pivot.columns]

        predict_pivot.loc["Totals"] = predict_pivot.sum()
        predict_pivot_styled = predict_pivot.style.format("{:,.0f}")

        # ----------------------------
        # Build yearly forecast totals
        # ----------------------------

        # Actuals by year
        actuals_by_year = (
            filtered_df
            .groupby(filtered_df['ds'].dt.year)
            .agg({"Current_Month_Actuals": "sum"})
            .rename(columns={"Current_Month_Actuals": "actuals"})
        )

        # Forecast by year (future only)
        forecast_by_year = (
            predict_df[predict_df['ds'] > sum_actuals['ds'].max()]
            .groupby(predict_df['ds'].dt.year)
            .agg({"yhat1": "sum"})
            .rename(columns={"yhat1": "forecast"})
        )

        # Combine actuals + forecast
        yearly_totals = actuals_by_year.join(forecast_by_year, how="outer").fillna(0)
        yearly_totals["total"] = yearly_totals["actuals"] + yearly_totals["forecast"]
        yearly_totals = yearly_totals.sort_index()
        yearly_totals["delta"] = yearly_totals["total"].diff()

        # Invert delta for Streamlit coloring (expenses: lower is better)
        yearly_totals["delta_display"] = -yearly_totals["delta"]

        # ----------------------------
        # Define hot take years
        # ----------------------------

        current_year = sum_actuals['ds'].dt.year.max()

        hot_take_years = [
            current_year,
            current_year + 1,
            current_year + 2,
            current_year + 3,
            current_year + 4,
            current_year + 5
        ]

        # ----------------------------
        # Combined Actuals + Forecast Table
        # ----------------------------
        with metrics_container:
            # ----------------------------
            # Forecast Hot Takes (TOP)
            # ----------------------------
            if forecast_change == 0:
                st.subheader("🔥 Forecast Hot Takes")
            else:
                st.subheader(f"🔥 Forecast Hot Takes {forecast_change:+,.2f}%"
                )

            tiles_per_row = 3

            for i in range(0, len(hot_take_years), tiles_per_row):
                cols = st.columns(tiles_per_row)

                for col, year in zip(cols, hot_take_years[i:i + tiles_per_row]):
                    if year in yearly_totals.index:
                        total_value = yearly_totals.loc[year, "total"]
                        delta_value = yearly_totals.loc[year, "delta"]

                        if pd.isna(delta_value):
                            delta_display = None
                        else:
                            delta_display = f"{delta_value:,.0f}"  # keeps minus sign + commas

                        col.metric(
                            label=f"FY {year}",
                            value=f"${total_value:,.0f}",
                            delta=delta_display,
                            border=True
                        )

        with table_container:
            if forecast_change == 0:
                st.subheader("Actuals & Forecast by Period and Fiscal Year")
            else: st.subheader(f"Actuals & Forecast by Period and Fiscal Year {forecast_change:+.2f}%"
                )

            CMA_pivot_prefixed = CMA_pivot.add_prefix("Actual_")
            predict_pivot_prefixed = predict_pivot.add_prefix("Forecast_")

            df_combined = pd.concat(
                [CMA_pivot_prefixed, predict_pivot_prefixed],
                axis=1
            )

            # Find the overlapping year (exists in both Actual_ and Forecast_ columns)
            # Remove decimals from column names
            df_combined.columns = [
                col.split("_")[0] + "_" + str(int(float(col.split("_")[1])))
                if "_" in col and col.split("_")[1].replace(".", "").isdigit()
                else col
                for col in df_combined.columns
            ]

            # Find overlapping years
            actual_years = [c.split("_")[1] for c in df_combined.columns if c.startswith("Actual_")]
            forecast_years = [c.split("_")[1] for c in df_combined.columns if c.startswith("Forecast_")]
            overlap_years = set(actual_years) & set(forecast_years)

            # Merge overlapping columns
            for year in overlap_years:
                actual_col = f"Actual_{year}"
                forecast_col = f"Forecast_{year}"
                df_combined[actual_col] = df_combined[actual_col].fillna(df_combined[forecast_col])
                df_combined.drop(columns=[forecast_col], inplace=True)

            CMA_pivot_prefixed = CMA_pivot.add_prefix("Actual_")

            df_combined.drop(index = ["Totals"], inplace=True)

            df_combined.loc["Totals"] = df_combined.sum()

            df_combined.index.name = "Period"

            #  Shade any year/period beyond maximum date in light gray

            def shade_forecast(column, max_dt):
                fiscal_to_calendar_month = {
                    1: 7, 2: 8, 3: 9, 4: 10, 5: 11, 6: 12,
                    7: 1, 8: 2, 9: 3, 10: 4, 11: 5, 12: 6
                }
                color = []
                fiscal_year = int(float(column.name.split("_")[1]))
                for i, period_element in enumerate(column.index):
                    if period_element == "Totals":
                        color.append(color[-1])
                        continue
                    try:
                        period_int = int(float(period_element))
                        calendar_month = fiscal_to_calendar_month[period_int]
                        # Periods 1-6 (Jul-Dec) are in previous calendar year
                        if period_int <= 6:
                            calendar_year = fiscal_year - 1
                        else:
                            calendar_year = fiscal_year
                        ds = pd.to_datetime(f"{calendar_year}-{calendar_month:02d}-01")
                        if ds > max_dt:
                            color.append("background-color: lightgray")
                        else:
                            color.append("background-color: white")
                    except (ValueError, TypeError, KeyError):
                        color.append("background-color: white")
                return color

            styled_combined = (
                df_combined
                .style
                .apply(shade_forecast, max_dt = max_dt)
                .format("{:,.0f}")
            )

            st.write(styled_combined)

            def convert_for_download(df):
                return df.to_csv().encode("utf-8")

            csv = convert_for_download(df_combined)

            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="data.csv",
                mime="text/csv",
                icon=":material/download:",
            )

# ----------------------------
# Run App
# ----------------------------

if __name__ == "__main__":
    main()