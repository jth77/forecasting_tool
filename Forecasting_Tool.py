import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from neuralprophet import NeuralProphet

# Configuration
DATA_PATH = Path(
    r"proxydataset.csv")
LOGO_PATH = Path(r"FSSLogo.png")


def handle_duplicate_columns(df):
    """Clean and deduplicate column names"""
    # Standardize column names
    df.columns = (
        df.columns.str.strip()
        .str.replace('\xa0', ' ')
        .str.replace(r'[^a-zA-Z0-9]', ' ', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
        .str.replace(' ', '_')
    )

    # Rename duplicate columns
    cols = pd.Series(df.columns)
    duplicates = cols[cols.duplicated()].unique()
    for dup in duplicates:
        cnt = 1
        for idx in cols[cols == dup].index:
            cols[idx] = f"{dup}_{cnt}"
            cnt += 1
    df.columns = cols
    return df


@st.cache_data
def load_data():
    """Load and preprocess data with error handling"""
    try:
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

        df = pd.read_csv(DATA_PATH, encoding='utf-8')
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

@st.cache_data
def run_fit_predict(input_df):
    m = NeuralProphet()

    metrics = m.fit(input_df)
    #

    # Create a new dataframe reaching 365 into the future for our forecast, n_historic_predictions also shows historic data
    df_future = m.make_future_dataframe(input_df, n_historic_predictions=True, periods=36)
    # n could be a user choice to choose how far back the history should go

    forecast = m.predict(df_future)
    print(forecast)
    my_plot = m.plot(forecast)
    # print(my_plot)
    # st.pyplot(my_plot)
    st.plotly_chart(my_plot)
    return forecast

def main():
    # App Configuration
    st.set_page_config(page_title="Financial Forecasting Tool", layout="wide")

    # Load data
    df = load_data()

    # Sidebar with Logo and Filters
    with st.sidebar:
        try:
            if LOGO_PATH.exists():
                st.image(str(LOGO_PATH), use_container_width=True)
            else:
                st.warning(f"Logo not found at: {LOGO_PATH}")
        except Exception as e:
            st.error(f"Logo loading error: {str(e)}")

        st.header("Data Filters")

        # Dynamic filter creation
        filter_columns = [
            'Organization_Code',
            'Account_Number',
            'Sub_Account_Number',
            'Object_Code',
            'Period_Number',
            'Fiscal_Year',
            'Category_Description'
        ]

        selected_filters = {}
        filter_types = ["Contains", "Starts with", "Ends with"]

        for col in filter_columns:
            try:
                options = df[col].dropna().unique()

                # Format Object Code columns to 4-digit strings
                if col in ["Object_Code"]:
                    options = [str(int(x)).zfill(4) if pd.notnull(x) and str(x).isdigit() else str(x) for x in options]

                # Attempt to sort numerically if possible; fall back to string sort
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
                st.error(f"Column '{col}' not found in dataset")
                st.stop()
    # print(selected_filters)

    # Apply filters
    filtered_df = df.copy()
    for col, values in selected_filters.items():
        if values:
            filtered_df = filtered_df[filtered_df[col].isin(values)]
    filtered_df.to_csv("proxydataset.csv",index=False)

    # Main Content
    col1, col2 = st.columns([1, 3])
    # print(filtered_df)

    # CMA = filtered_df["Current_Month_Actuals"]
    #     # FY = filtered_df["Fiscal_Year"]
    #     # FY_2023 = FY == 2023
    #     # CMA_2023 = CMA[FY_2023]
    #     # CMA_2023.sum()
    #     #
    #     # # filtering for period 1
    #     # PER = filtered_df["Period_Number"]
    #     # PER_01 = PER == "01"
    #     # CMA_01 = CMA[PER_01]
    #     # PER_01.sum()

    periods_sum = (
    filtered_df
        .groupby(["Fiscal_Year", "Period_Number"], as_index = False)
        .agg({"Current_Month_Actuals": "sum"})
    )

    CMA_pivot = periods_sum.pivot(
                      index = 'Period_Number',
                      columns = 'Fiscal_Year',
                      values = 'Current_Month_Actuals')

    CMA_pivot_styled = CMA_pivot.style.format(lambda x: f"{x:,.0f}")

    st.write("Period Actuals")
    st.dataframe(CMA_pivot_styled, use_container_width=False)

    # df = pd.read_csv('toiletpaper_daily_sales.csv')

    filtered_df = filtered_df.drop(filtered_df[filtered_df.Period_Number == "BB"].index)
    filtered_df = filtered_df.drop(filtered_df[filtered_df.Period_Number == "CB"].index)
    # print(filtered_df.dtypes)
    #     filtered_df[filtered_df["Period_Number"]=="13"] = 12
    filtered_df.Period_Number[filtered_df["Period_Number"] == "13"] = 12

    filtered_df['date_string'] = filtered_df['Fiscal_Year'].astype(str) + '-' + filtered_df['Period_Number'].astype(
        str) + '-01'
    # print(filtered_df['date_string'])
    # print(filtered_df.where(filtered_df["date_string"] == "12-12-01"))
    filtered_df['ds'] = pd.to_datetime(filtered_df['date_string'])
    print("made it to here")
    # make new dataframe (select rows with year 2025)
    # then make use of groupby techniques (higher up), so sum

    # #print(filtered_df)
    sum_actuals = filtered_df.groupby('ds', as_index=False).agg({"Current_Month_Actuals": 'sum'})
    # # st.table(data=sum_actuals.iloc[:200])
    sum_actuals['y'] = sum_actuals.Current_Month_Actuals
    # # print(sum_actuals.Current_Month_Actuals)
    sum_actuals = sum_actuals.drop(columns='Current_Month_Actuals')
    predict_df = run_fit_predict(sum_actuals)

    # Extract the year and month into a new column
    predict_df['year'] = predict_df['ds'].dt.year
    predict_df['month'] = predict_df['ds'].dt.month
    print(predict_df)

    # CMA_pivot = periods_sum.pivot(
    #                   index = 'Period_Number',
    #                   columns = 'Fiscal_Year',
    #                   values = 'Current_Month_Actuals')
    #
    # CMA_pivot_styled = CMA_pivot.style.format(lambda x: f"{x:,.0f}")
    #
    # st.write("Period Actuals")
    # st.dataframe(CMA_pivot_styled, use_container_width=False)

if __name__ == "__main__":
    main()

