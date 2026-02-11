# **FSS Forecasting Tool**

##### The FSS Forecasting Tool is an interactive forecasting dashboard built with Streamlit and NeuralProphet. It allows users to upload fiscal data, apply dynamic filters, and visualize time-series forecasts with adjustable percentage changes. The app produces period-level and yearly projections for actuals and forecasts with clear visuals and downloadable results.

# 

## Features

* ##### 📁 CSV Upload: Upload fiscal data containing organization and account-level metrics.
* ##### 🧹 Automatic Data Cleaning: Removes duplicates, normalizes column names, and validates required fields.
* ##### ⚙️ Flexible Filtering: Filter data interactively by organization, account, sub-account, and other attributes.
* ##### 🔮 Forecast Generation: Uses NeuralProphet to forecast financial metrics over upcoming months.
* ##### 📊 Visual Dashboard: Displays time-series plots, forecast summaries, and yearly “hot take” metrics.
* ##### 💾 Data Export: Download combined actuals and forecasted values as a CSV file.

# 

### Installation

##### Make sure you have Python 3.9+ installed. Then, set up the environment and install required dependencies:

##### 

##### pip install streamlit pandas neuralprophet plotly

##### 

##### If neuralprophet requires additional torch dependencies, install them as prompted.

##### 

### Usage

1. Place your fiscal data CSV file in the project directory.
   The file must include the following columns:
   - Organization\_Code
   - Account\_Number
   - Fiscal\_Year
   - Current\_Month\_Actuals
   ---
2. ##### (Optional) Include FSSLogo.png in the same directory to display your logo in the sidebar.
3. Run the app with:
   - streamlit run app.py
   ---
4. Once launched:
   - Upload your fiscal data file.
   - Apply filters from the sidebar.
   - Adjust the “Forecast (%)” input to manually increase or decrease predicted values.
   - View plots, yearly summaries, and detailed tables within the main interface.
   - Export results by clicking Download CSV.
   ---

##### 

### How It Works

* ##### Data Preparation: Uploaded CSVs are cleaned and validated for missing or duplicate fields.
* ##### Time Series Construction: Fiscal periods are converted to monthly timestamps (ds).
* ##### Model Fitting: NeuralProphet fits the actuals (y) and generates a 62-period forward forecast.
* ##### Visualization: Forecasts and actuals are displayed with interactive Plotly charts.
* ##### Summary Metrics: Yearly totals and deltas between years are shown as Streamlit metric tiles.

##### 

### File Structure

##### text

##### .

##### ├── app.py              # Main Streamlit application

##### ├── FSSLogo.png         # Optional sidebar logo

##### ├── README.md           # Documentation (this file)

##### └── requirements.txt    # Dependencies list

# 

### Example Data Format

|Organization\_Code|Account\_Number|Fiscal\_Year|Period\_Number|Current\_Month\_Actuals|
|-|-|-|-|-|
|1001|200345|2023|01|12500.00|
|1001|200345|2023|02|13200.00|
|...|...|...|...|...|

# 

				

License

This project is released under the MIT License.

Feel free to modify, distribute, and adapt it to your organization’s forecasting needs.

