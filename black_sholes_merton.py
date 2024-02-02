import numpy as np
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
  
'''
# Example usage
S = 100  # Current stock price
K = 100  # Strike price
T = 1    # Time to maturity (in years)
r = 0.05 # Risk-free rate
market_price = 10  # Observed market price of the option

# Calculate implied volatility
sigma_imp = implied_volatility(market_price, S, K, T, r)
print(f"Implied Volatility: {sigma_imp}")

# Example historical prices (daily closing prices)
prices = np.random.normal(loc=100, scale=10, size=252)  # Example data
sigma_hist = historical_volatility(prices)
print(f"Historical Volatility: {sigma_hist}")
'''

# Example of reading processed data
price_data = pd.read_csv('historical_prices.csv')  # Assuming a CSV with a 'Close' column
option_data = pd.read_csv('option_market_data.csv')  # Assuming CSV with 'MarketPrice', 'Strike', 'Maturity'

# Processing historical volatility
historical_vol = historical_volatility(price_data['Close'])

# Processing implied volatility for each option in the dataset
option_data['ImpliedVolatility'] = option_data.apply(
    lambda row: implied_volatility(row['MarketPrice'], S, row['Strike'], row['Maturity'], r),
    axis=1
)

print(option_data)
