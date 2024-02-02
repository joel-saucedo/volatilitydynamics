import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot as plt

def test_stationarity(timeseries):
    """Perform Dickey-Fuller test to assess stationarity"""
    result = adfuller(timeseries, autolag='AIC')
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

def prepare_data(df, column_name):
    """Prepare and clean high-frequency data for analysis"""
    # Convert index to datetime if necessary
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Log transform for stationarity
    df[column_name] = np.log(df[column_name])
    return df[column_name]

def arima_model(timeseries):
    """Fit an ARIMA model to the timeseries"""
    model = ARIMA(timeseries, order=(5,1,0)) # Example order, adjust based on ACF and PACF plots
    model_fit = model.fit()
    print(model_fit.summary())
    
    # Plot the residuals to check for any patterns
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    plt.show()
    residuals.plot(kind='kde')
    plt.show()
    print(residuals.describe())

# Example usage
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('high_frequency_data.csv') # Ensure your CSV file has 'Date' and the column you're interested in
    column_name = 'Price' # Change according to your data
    
    timeseries = prepare_data(df, column_name)
    
    # Test for stationarity
    test_stationarity(timeseries)
    
    # If not stationary, differencing or transformation might be needed
    # For simplicity, we're moving forward assuming stationarity or after making the series stationary
    
    # Fit ARIMA model
    arima_model(timeseries)
