import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate the Black-Scholes option price for a European call option.
    S: Current stock price
    K: Strike price
    T: Time to maturity
    r: Risk-free rate
    sigma: Volatility of the stock
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def implied_volatility(market_price, S, K, T, r):
    """
    Calculate implied volatility by finding the sigma that minimizes
    the difference between the market call price and the Black-Scholes call price.
    """
    def objective(sigma): return market_price - black_scholes_call(S, K, T, r, sigma)
    return brentq(objective, 0.0001, 2.0)

def historical_volatility(prices):
    """
    Calculate historical volatility from asset prices.
    """
    log_returns = np.diff(np.log(prices))
    sigma_hist = np.std(log_returns) * np.sqrt(252)  # Annualize
    return sigma_hist

# Load stock data
stock_data = pd.read_csv('historical_prices.csv')
# Calculate historical volatility
historical_vol = historical_volatility(stock_data['Close'])

# Load options data
options_data = pd.read_csv('option_market_data.csv')

# Assuming you have specific values for S (current stock price) and r (risk-free rate)
# These could be constants or derived from another part of your analysis
S = stock_data['Close'].iloc[-1]  # Example: using the last available stock price
r = 0.05  # Example: a given risk-free rate, you may need to adjust this based on current rates

# Calculate implied volatility for each option
options_data['ImpliedVolatility'] = options_data.apply(
    lambda row: implied_volatility(row['MarketPrice'], S, row['Strike'], row['Maturity'] / 365, r),
    axis=1  # Ensure 'Maturity' is in years; adjust if your data is in days
)

print(f"Historical Volatility: {historical_vol}")
print(options_data[['MarketPrice', 'Strike', 'Maturity', 'ImpliedVolatility']])
