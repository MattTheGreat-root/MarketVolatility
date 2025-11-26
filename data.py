import yfinance as yf
ticker = "AAPL"
df = yf.download(ticker, start="2014-01-01", end="2024-01-01")
df.to_csv("AAPL_2014_224.csv")