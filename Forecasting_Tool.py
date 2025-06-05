import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Configuration
DATA_PATH = Path(
    r"C:\Users\jtherman.BLUECAT\OneDrive - University of Arizona\Projects\Forecasting_Tool\Account Detail_ Transactions.csv")
LOGO_PATH = Path(r"C:\Users\jtherman.BLUECAT\OneDrive - University of Arizona\Projects\Forecasting_Tool\FSSLogo.png")


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
            'Sub_Object_Code',
            'Project_Code',
            'Period_Number'
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

    # Apply filters
    filtered_df = df.copy()
    for col, values in selected_filters.items():
        if values:
            filtered_df = filtered_df[filtered_df[col].isin(values)]

    # Main Content
    col1, col2 = st.columns([1, 3])

    with col1:
        st.header("Data Summary")
        if not filtered_df.empty:
            st.metric("Total Records", len(filtered_df))
            st.metric("Unique Fiscal Years", filtered_df["Fiscal_Year"].nunique())

            with st.expander("Sample Data"):
                st.dataframe(
                    filtered_df.head(10),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.warning("No data matches selected filters")

    with col2:
        st.header("Trend Analysis")

        if not filtered_df.empty:
            try:
                agg_df = filtered_df.groupby("Fiscal_Year", as_index=False).agg({
                    "Current_Month_Actuals": "sum"
                })

                # Ensure Fiscal Year is treated as categorical data to fix x-axis issue
                agg_df["Fiscal_Year"] = agg_df["Fiscal_Year"].astype(str)

                fig = px.line(
                    agg_df,
                    x="Fiscal_Year",
                    y="Current_Month_Actuals",
                    title="Actual Expenses Over Time",
                    markers=True,
                    category_orders={"Fiscal_Year": sorted(agg_df["Fiscal_Year"].unique())}
                    # Sort fiscal years properly
                )

                # Explicitly set x-axis type to "category"
                fig.update_layout(
                    xaxis_type="category",
                    hovermode="x unified",
                    yaxis_tickprefix="$",
                    height=600,
                    xaxis_title="Fiscal Year",
                    yaxis_title="Current Month Actuals"
                )

                st.plotly_chart(fig, use_container_width=True)

            except KeyError as e:
                st.error(f"Missing required column: {str(e)}")
        else:
            st.info("Select filters to view trends")


if __name__ == "__main__":
    main()
    