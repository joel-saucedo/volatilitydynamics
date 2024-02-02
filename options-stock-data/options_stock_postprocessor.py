import pandas as pd
import numpy as np

# Function to read and process stock data
def process_stock_data(file_path):
    stock_data = pd.read_csv(file_path)
    
    # If necessary, perform any transformations or cleaning on stock_data
    # For this example, we assume 'Close' is the column we need directly
    
    # Save processed stock data
    stock_data.to_csv('historical_prices.csv', index=False)

# Function to read and process options data
def process_options_data(file_path):
    options_data = pd.read_csv(file_path)
    
    # Convert 'expiration' to 'Maturity' in years (assuming 'expiration' is in YYYY-MM-DD format)
    options_data['quote_date'] = pd.to_datetime(options_data['quote_date'])
    options_data['expiration'] = pd.to_datetime(options_data['expiration'])
    options_data['Maturity'] = (options_data['expiration'] - options_data['quote_date']).dt.days / 365
    
    # Assuming 'MarketPrice' can be approximated with the mid-point of 'bid' and 'ask'
    options_data['MarketPrice'] = (options_data['bid'] + options_data['ask']) / 2
    
    # Assuming 'Strike' is already in the correct format
    # If 'type' is 'call' or 'put', ensure it matches your analysis requirements
    
    # Select and rename columns to match expected input format
    processed_options_data = options_data[['MarketPrice', 'strike', 'Maturity']]
    processed_options_data.rename(columns={'strike': 'Strike'}, inplace=True)
    
    # Save processed options data
    processed_options_data.to_csv('option_market_data.csv', index=False)

# Assuming the original files are named '2013-01-02stocks.csv' and '2013-01-02options.csv'
process_stock_data('2013-01-02stocks.csv')
process_options_data('2013-01-02options.csv')

print("Data processing complete. Files 'historical_prices.csv' and 'option_market_data.csv' are ready.")
