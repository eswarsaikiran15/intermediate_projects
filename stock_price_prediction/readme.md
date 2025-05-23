# Stock Price Prediction with Time Series

## Overview

This project predicts future stock prices for Apple (AAPL) using historical price and volume data from Yahoo Finance. It implements three time-series forecasting models: **ARIMA**, **LSTM**, and **Prophet**, incorporating preprocessing, feature engineering (e.g., moving averages), and evaluation with metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). The project also visualizes actual vs. predicted prices.

**Note**: Stock price prediction is inherently uncertain due to market volatility and external factors. This project is for educational purposes and should not be used for financial decisions.

## Features

- **Data Source**: Historical stock data (2010–2025) from Yahoo Finance using `yfinance`.
- **Preprocessing**: Handles missing values and creates lag features (e.g., previous day's closing price).
- **Feature Engineering**: Computes technical indicators like the 50-day moving average (MA50).
- **Models**:
    - **ARIMA**: Statistical model for stationary time series.
    - **LSTM**: Deep learning model for capturing long-term patterns.
    - **Prophet**: Meta’s forecasting model for non-stationary data with seasonality.
- **Evaluation**: Uses MAE and RMSE to assess model performance.
- **Visualization**: Plots actual vs. predicted prices, saved as `stock_predictions.png`.

## Requirements

- Python 3.7–3.10 (Prophet compatibility)
- Dependencies:
    
    ```bash
    pip install yfinance prophet tensorflow statsmodels scikit-learn pandas numpy matplotlib pystan cmdstanpy
    ```
    

## Setup

1. **Clone or Download the Project**:
    
    - Save the `stock_price_prediction.py` script in a project directory.
2. **Install Dependencies**:  
    Run the following command in your terminal or Jupyter Notebook:
    
    ```bash
    pip install yfinance prophet tensorflow statsmodels scikit-learn pandas numpy matplotlib pystan cmdstanpy
    ```
    
3. **Ensure Internet Connectivity**:
    
    - The `yfinance` library requires an internet connection to download stock data.

## Usage

1. **Run the Script**:  
    Execute the Python script in your environment:
    
    ```bash
    python stock_price_prediction.py
    ```
    
    Alternatively, in Jupyter Notebook or Google Colab:
    
    - Copy and paste the code into a cell.
    - Run the cell to execute.
2. **Expected Output**:
    
    - Console output:
        - MAE and RMSE for ARIMA, LSTM, and Prophet models (e.g., `LSTM - MAE: 13.04, RMSE: 14.70`).
        - Debug information for Prophet (e.g., DataFrame head, shapes, and data types).
    - A plot (`stock_predictions.png`) comparing actual and predicted prices.
    - If any model fails, an error message will be printed, and the plot will exclude that model’s predictions.
3. **File Structure**:
    
    ```
    project_directory/
    ├── stock_price_prediction.py
    ├── stock_predictions.png (generated after running)
    └── README.md
    ```
    

## Code Structure

- **Data Acquisition**: Downloads AAPL stock data using `yfinance`.
- **Preprocessing**: Removes missing values and adds lag features.
- **Feature Engineering**: Computes a 50-day moving average (MA50).
- **Model Training**:
    - ARIMA: Uses order `(1, 1, 1)` (can be optimized with `pmdarima`).
    - LSTM: Uses a 60-day sequence length with two features (Close, MA50).
    - Prophet: Fits a model with daily seasonality.
- **Evaluation**: Computes MAE and RMSE for each model.
- **Visualization**: Plots actual vs. predicted prices with Matplotlib.

## Example Output

- Console (sample):
    
    ```
    ARIMA - MAE: 5.23, RMSE: 7.15
    LSTM - MAE: 13.04, RMSE: 14.70
    Prophet DataFrame head:
               ds         y
    0 2010-03-16  29.37
    1 2010-03-17  29.60
    ...
    Shape of df_prophet['y']: (3821,)
    Type of df_prophet['y']: float64
    Type of df_prophet['ds']: datetime64[ns]
    Prophet - MAE: 6.12, RMSE: 8.34
    ```
    
- Plot: A line graph showing actual prices (blue), ARIMA (green), LSTM (red), and Prophet (orange) predictions.

## Troubleshooting

- **Prophet Errors**:
    
    - If Prophet fails (e.g., `Per-column arrays must each be 1-dimensional`):
        - Check debug output for `df_prophet` (shape and dtypes).
        - Reinstall dependencies:
            
            ```bash
            pip install pystan cmdstanpy prophet --force-reinstall
            ```
            
        - Ensure Python 3.7–3.10.
    - Test Prophet with a minimal dataset:
        
        ```python
        df_test = pd.DataFrame({
            'ds': pd.date_range(start='2020-01-01', periods=100),
            'y': np.random.rand(100)
        })
        model_test = Prophet().fit(df_test)
        ```
        
- **yfinance Errors**:
    
    - Ensure internet connectivity.
    - Update `yfinance`:
        
        ```bash
        pip install yfinance --upgrade
        ```
        
- **Plotting Issues**:
    
    - Verify `matplotlib` is installed:
        
        ```bash
        pip install matplotlib
        ```
        
    - Check that prediction arrays align with `test_index`.

## Potential Improvements

- **Additional Features**:
    
    - Add technical indicators like RSI or MACD:
        
        ```python
        def calculate_rsi(data, periods=14):
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        data['RSI'] = calculate_rsi(data)
        ```
        
        Update LSTM input to include `RSI`.
- **Model Tuning**:
    
    - Optimize ARIMA parameters:
        
        ```bash
        pip install pmdarima
        ```
        
        ```python
        from pmdarima import auto_arima
        model_arima = auto_arima(train, seasonal=False, trace=True)
        ```
        
    - Tune LSTM (e.g., increase epochs, add layers, adjust sequence length).
- **Data Range**:
    
    - Modify the date range in `yf.download` for different periods or stocks.

## Limitations

- Stock price prediction is highly uncertain due to market volatility and external factors (e.g., economic conditions, news).
- The models provide insights but are not reliable for trading decisions.
- ARIMA assumes stationary data, which stock prices often aren’t.
- LSTM requires careful tuning to avoid overfitting.
- Prophet may struggle with rapid price changes.

## Resources

- [Prophet Documentation](https://facebook.github.io/prophet/docs/quick_start.html)
- [yfinance GitHub](https://github.com/ranaroussi/yfinance)
- [TensorFlow LSTM Tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Statsmodels ARIMA Documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)

## License

This project is for educational purposes and is provided as-is. No warranty is implied for financial use.

## Contact

For issues or questions, please check the troubleshooting section or consult the referenced documentation.
