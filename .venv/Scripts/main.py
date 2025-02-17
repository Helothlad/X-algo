import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
import os
import matplotlib.ticker as mtick  # <-- Add this import

plt.style.use('ggplot')

data_folder = 'C:/Users/peper/PycharmProjects/Twitter-Sentiment-Investing-Strategy/.venv'

# Read the CSV
sentiment_df = pd.read_csv(os.path.join(data_folder, 'sentiment_data.csv'))

# Ensure the date is in datetime format
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
sentiment_df = sentiment_df.set_index(['date', 'symbol'])

# Calculate engagement ratio
sentiment_df['engagement_ratio'] = sentiment_df['twitterComments'] / sentiment_df['twitterLikes']
sentiment_df = sentiment_df[(sentiment_df['twitterLikes'] > 20) & (sentiment_df['twitterComments'] > 10)]

# Aggregate sentiment data by month and calculate rankings
aggragated_df = (sentiment_df.reset_index('symbol').groupby([pd.Grouper(freq='ME'), 'symbol'])
                    [['engagement_ratio']].mean())
aggragated_df['rank'] = (aggragated_df.groupby(level=0)['engagement_ratio']
                         .transform(lambda x: x.rank(ascending=False)))

# Filter for top-ranked stocks
filtered_df = aggragated_df[aggragated_df['rank'] < 6].copy()
filtered_df = filtered_df.reset_index(level=1)
filtered_df.index = filtered_df.index + pd.DateOffset(1)
filtered_df = filtered_df.reset_index().set_index(['date', 'symbol'])

# Extract dates
dates = filtered_df.index.get_level_values('date').unique().tolist()
fixed_dates = {}
for d in dates:
    fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level=0).index.tolist()

# Get the list of stocks
stocks_list = sentiment_df.index.get_level_values('symbol').unique().tolist()

# Download stock price data
prices_df = yf.download(tickers=stocks_list,
                        start='2021-01-01',
                        end='2023-03-01')

# Debug: Print available columns to check what columns are available
print(prices_df.columns)

# If 'Adj Close' is missing, use 'Close'
returns_df = np.log(prices_df['Close']).diff().dropna()

# Create portfolio returns
portfolio_df = pd.DataFrame()
for start_date in fixed_dates.keys():
    end_date = (pd.to_datetime(start_date) + pd.offsets.MonthEnd()).strftime('%Y-%m-%d')
    cols = fixed_dates[start_date]
    temp_df = returns_df[start_date:end_date][cols].mean(axis=1).to_frame('portfolio_return')
    portfolio_df = pd.concat([portfolio_df, temp_df], axis=0)

qqq_df = yf.download(tickers='QQQ',
                     start='2021-01-01',
                     end='2023-03-01')

qqq_ret = np.log(qqq_df['Close']).diff()
portfolio_df = portfolio_df.merge(qqq_ret,
                                  left_index=True,
                                  right_index=True)

portfolios_cumulative_return = np.exp(np.log1p(portfolio_df).cumsum()).sub(1)
portfolios_cumulative_return.plot(figsize=(16,6))
plt.title('Twitter Engagement Ratio Strategy Return Over Time')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))  # <-- This line now works because of the import
plt.ylabel('Return')
plt.show()
