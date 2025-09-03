# =======================================================================
# Bitcoin Trading Strategy (Last 12 Months)
# Moving Average Crossover + Backtest
# =======================================================================

from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

# ========================================================================
# Step 1: Download BTC-USD data (last 12 months)
# ========================================================================

# Get today's date
today = datetime.today()

# Get date one year ago from today
one_year_ago = today - timedelta(days=365)

# Download daily Bitcoin price data from Yahoo Finance
BTC_USD = yf.download(
    "BTC-USD",
    start=one_year_ago.strftime("%Y-%m-%d"),
    end=today.strftime("%Y-%m-%d"),
    interval="1d"
)

# =======================================================================
# Step 2: Create trading signals
# =======================================================================

trade_signals = pd.DataFrame(index=BTC_USD.index)

# Short and long moving average windows
short_interval = 10   # short-term average (10 days)
long_interval = 40    # long-term average (40 days)

# Calculate moving averages of Bitcoin closing price
trade_signals['Short'] = BTC_USD['Close'].rolling(window=short_interval, min_periods=1).mean()
trade_signals['Long'] = BTC_USD['Close'].rolling(window=long_interval, min_periods=1).mean()

# Signal = 1 if short MA > long MA, else 0
trade_signals['Signal'] = np.where(trade_signals['Short'] > trade_signals['Long'], 1.0, 0.0)

# Position = Change in Signal (Buy=1, Sell=-1, Hold=0)
trade_signals['Position'] = trade_signals['Signal'].diff()

# =================================================================================
# Step 3: Plot Strategy Chart
# =================================================================================

fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

# Format x-axis to show month-year (Jan-24, Feb-24, …)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(DateFormatter("%b-%y"))
fig.autofmt_xdate()

# Plot Bitcoin closing price
ax.plot(BTC_USD['Close'], lw=1, label='Closing Price', color="blue")

# Plot short & long SMAs
ax.plot(trade_signals['Short'], lw=1, alpha=0.9, color='orange', label=f'{short_interval}-Day SMA')
ax.plot(trade_signals['Long'], lw=1, alpha=0.9, color='purple', label=f'{long_interval}-Day SMA')

# Buy signals (green arrows)
ax.scatter(
    trade_signals.loc[trade_signals['Position'] == 1.0].index,
    trade_signals['Short'][trade_signals['Position'] == 1.0],
    marker="^", s=80, color='green', label="Buy Signal"
)

# Sell signals (red arrows)
ax.scatter(
    trade_signals.loc[trade_signals['Position'] == -1.0].index,
    trade_signals['Short'][trade_signals['Position'] == -1.0],
    marker="v", s=80, color='red', label="Sell Signal"
)

# Labels and title
ax.set_ylabel("Price of Bitcoin (USD)")
ax.set_title("Bitcoin Moving Average Crossover Strategy (Last 12 Months)")

# Grid + legend
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=9, frameon=True)

# Show chart
plt.show()

# ===================================================================================
# Step 4: Backtesting
# ===================================================================================

initial_balance = 10000.0  # Start with $10,000

# Create dataframe to hold backtest results
backtest = pd.DataFrame(index=trade_signals.index)

# Daily returns of Bitcoin (ratio, not %)
backtest['BTC_Return'] = BTC_USD['Close'] / BTC_USD['Close'].shift(1)

# Algorithm returns: if in position (Signal=1), follow BTC return; otherwise hold USD (return=1.0)
backtest['Alg_Return'] = np.where(trade_signals['Signal'] == 1, backtest['BTC_Return'], 1.0)

# Calculate portfolio values
backtest['Balance'] = initial_balance * backtest['Alg_Return'].cumprod()  # Strategy balance
backtest['BuyHold'] = initial_balance * backtest['BTC_Return'].cumprod()  # Buy & Hold balance

# ==================================================================================
# Step 5: Plot Backtest Results
# ==================================================================================

fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

# Format x-axis
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(DateFormatter("%b-%y"))
fig.autofmt_xdate()

# Plot Buy & Hold vs Strategy
ax.plot(backtest['BuyHold'], lw=1, alpha=0.9, label='Buy & Hold', color="blue")
ax.plot(backtest['Balance'], lw=1, alpha=0.9, label='Moving Average Crossover', color="orange")

# Labels and title
ax.set_ylabel("Portfolio Value (USD)")
ax.set_title("Backtest of Bitcoin Strategy vs Buy & Hold (Last 12 Months)")

# Grid + legend
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=9, frameon=True)

# Show chart
plt.show()

# ========================================================================================
# Step 6: Print Results
# ========================================================================================

print("Final Portfolio Value (Buy & Hold): $", round(backtest['BuyHold'].iloc[-1], 2))
print("Final Portfolio Value (Crossover): $", round(backtest['Balance'].iloc[-1], 2))




