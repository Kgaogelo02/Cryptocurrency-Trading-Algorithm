# Project inspired by:
# Cognitive Class — IBM GPXX0PICEN

import streamlit as st
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

st.set_page_config(page_title="BTC MA Crossover", layout="wide")

st.title("Bitcoin Moving Average Crossover — Last 12 Months")
st.markdown("**Project inspired by:** Cognitive Class — IBM GPXX0PICEN")

# Sidebar controls
st.sidebar.header("Settings")
short_interval = st.sidebar.slider("Short SMA (days)", min_value=2, max_value=50, value=10, step=1)
long_interval  = st.sidebar.slider("Long SMA (days)",  min_value=10, max_value=200, value=40, step=1)
initial_balance = st.sidebar.number_input("Initial balance (USD)", value=10000.0, step=100.0)
refresh_button = st.sidebar.button("Refresh data / Run")

# Auto-run (last 12 months)
today = datetime.today()
one_year_ago = today - timedelta(days=365)

@st.cache_data(ttl=300)
def download_data():
    df = yf.download("BTC-USD", start=one_year_ago.strftime("%Y-%m-%d"),
                     end=today.strftime("%Y-%m-%d"), interval="1d", progress=False)
    return df

BTC_USD = download_data()

if BTC_USD.empty:
    st.error("No data downloaded. Check internet connection or yfinance availability.")
    st.stop()

# Build signals
trade_signals = pd.DataFrame(index=BTC_USD.index)
trade_signals['Short'] = BTC_USD['Close'].rolling(window=short_interval, min_periods=1).mean()
trade_signals['Long']  = BTC_USD['Close'].rolling(window=long_interval,  min_periods=1).mean()
trade_signals['Signal'] = np.where(trade_signals['Short'] > trade_signals['Long'], 1.0, 0.0)
trade_signals['Position'] = trade_signals['Signal'].diff()

# Backtest
backtest = pd.DataFrame(index=trade_signals.index)
backtest['BTC_Return'] = BTC_USD['Close'] / BTC_USD['Close'].shift(1)
backtest['Alg_Return'] = np.where(trade_signals['Signal'] == 1, backtest['BTC_Return'], 1.0)
backtest['Balance'] = initial_balance * backtest['Alg_Return'].cumprod()
backtest['BuyHold'] = initial_balance * backtest['BTC_Return'].cumprod()

# Compute trades stats
def compute_trades_stats(signals, prices):
    trades = []
    current_buy = None
    for idx, row in signals.iterrows():
        pos = row['Position']
        if pos == 1.0:  # buy
            current_buy = prices.loc[idx, 'Close'].item()  # ensure scalar
        elif pos == -1.0 and current_buy is not None:  # sell
            sell_price = prices.loc[idx, 'Close'].item()  # ensure scalar
            trades.append((current_buy, sell_price))
            current_buy = None
    # If still holding at the end
    if current_buy is not None:
        trades.append((current_buy, prices['Close'].iloc[-1]))

    returns = [ (s/b - 1.0) for (b, s) in trades ] if trades else []
    wins = [r for r in returns if r > 0]
    win_rate = (len(wins) / len(returns)) * 100 if returns else np.nan
    avg_return = np.mean(returns) if returns else np.nan
    num_trades = len(returns)
    return {
        "num_trades": num_trades,
        "win_rate_pct": round(win_rate, 2) if not np.isnan(win_rate) else None,
        "avg_return_pct": round(avg_return*100, 2) if not np.isnan(avg_return) else None,
        "returns": returns
    }

stats = compute_trades_stats(trade_signals, BTC_USD)

# Max drawdown
running_max = backtest['Balance'].cummax()
drawdown = (running_max - backtest['Balance']) / running_max
max_drawdown = drawdown.max()

# Layout: charts and metrics
col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%b-%y"))
    fig.autofmt_xdate()

    ax.plot(BTC_USD['Close'], label='Closing Price', lw=1, color='tab:blue')
    ax.plot(trade_signals['Short'], label=f'{short_interval}-day SMA', lw=1, color='orange')
    ax.plot(trade_signals['Long'], label=f'{long_interval}-day SMA', lw=1, color='purple')

    buys_idx = trade_signals.loc[trade_signals['Position'] == 1.0].index
    sells_idx = trade_signals.loc[trade_signals['Position'] == -1.0].index
    ax.scatter(buys_idx, BTC_USD.loc[buys_idx, 'Close'], marker='^', s=80, color='green', label='Buy')
    ax.scatter(sells_idx, BTC_USD.loc[sells_idx, 'Close'], marker='v', s=80, color='red', label='Sell')

    ax.set_title("Price + SMAs (Last 12 Months)")
    ax.set_ylabel("USD")
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left')
    st.pyplot(fig)

with col2:
    st.subheader("Backtest Summary")
    st.markdown(f"- **Initial balance:** ${initial_balance:,.2f}")
    st.markdown(f"- **Final (Buy & Hold):** ${backtest['BuyHold'].iloc[-1]:,.2f}")
    st.markdown(f"- **Final (Strategy):** ${backtest['Balance'].iloc[-1]:,.2f}")
    st.markdown(f"- **Total trades (closed):** {stats['num_trades']}")
    st.markdown(f"- **Win rate:** {stats['win_rate_pct']} %")
    st.markdown(f"- **Avg trade return:** {stats['avg_return_pct']} %")
    st.markdown(f"- **Max drawdown (strategy):** {round(max_drawdown*100, 2)} %")

st.markdown("---")

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.xaxis.set_major_locator(mdates.MonthLocator())
ax2.xaxis.set_major_formatter(DateFormatter("%b-%y"))
fig2.autofmt_xdate()
ax2.plot(backtest['BuyHold'], label='Buy & Hold', lw=1, color='tab:blue')
ax2.plot(backtest['Balance'], label='Strategy', lw=1, color='orange')
ax2.set_title("Portfolio Value (Last 12 Months)")
ax2.set_ylabel("USD")
ax2.grid(alpha=0.3)
ax2.legend(loc='upper left')
st.pyplot(fig2)

# Optional trade table
if st.checkbox("Show trade signals table"):
    display_df = trade_signals[['Short', 'Long', 'Signal', 'Position']].copy()
    st.dataframe(display_df.tail(200))

st.markdown("**Notes:** This is an educational demo. No commission, slippage, or execution latency are modeled. Always forward-test in paper mode before real trading.")
