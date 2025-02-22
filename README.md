﻿# Twitter-sentiment-investing-strategy
This project implements an investing strategy based on Twitter engagement metrics (likes and comments) to identify and invest in stocks with high social media activity. The strategy ranks stocks monthly by their engagement ratio (comments divided by likes) and constructs a portfolio of the top-ranked stocks. The performance of this portfolio is then compared to the QQQ ETF (Nasdaq-100 Index) to evaluate its effectiveness.

How It Works:
Data Preparation:

Twitter sentiment data is loaded from a CSV file, containing metrics like twitterComments and twitterLikes for various stocks.

The data is filtered to include only stocks with sufficient engagement (e.g., more than 20 likes and 10 comments).

The engagement ratio is calculated as twitterComments / twitterLikes.

Monthly Ranking:

The sentiment data is aggregated monthly, and stocks are ranked based on their average engagement ratio.

The top 5 stocks with the highest engagement ratio are selected for each month.

Portfolio Construction:

Historical price data for the selected stocks is downloaded using the yfinance library.

Portfolio returns are calculated by equally weighting the returns of the top-ranked stocks each month.

Performance Comparison:

The portfolio's cumulative returns are computed and compared to the returns of the QQQ ETF.

Results are visualized using a cumulative return plot, with returns displayed as percentages.

Visualization:

The cumulative returns of the Twitter engagement strategy and the QQQ ETF are plotted to assess the strategy's performance over time.

Key Features:
Engagement Ratio: A novel metric combining Twitter comments and likes to gauge social media activity.

Monthly Rebalancing: The portfolio is rebalanced monthly based on the latest engagement data.

Benchmark Comparison: The strategy's performance is compared to the QQQ ETF, a proxy for the tech-heavy Nasdaq-100 Index.

Simulated Data: Demonstrates the strategy's potential using historical price and Twitter engagement data.

Dependencies:
pandas: For data manipulation and analysis.

numpy: For mathematical calculations.

matplotlib: For visualizing results.

yfinance: For downloading stock price data.

Usage:
Ensure the required libraries are installed.

Place the sentiment_data.csv file in the specified folder.

Run the script to:

Calculate engagement ratios and rank stocks.

Construct and rebalance the portfolio.

Compare the strategy's performance to the QQQ ETF.

Visualize the cumulative returns.

Example Output:
The script generates a plot showing the cumulative returns of the Twitter engagement strategy versus the QQQ ETF. This allows users to assess whether the strategy outperforms the benchmark over the tested period.
