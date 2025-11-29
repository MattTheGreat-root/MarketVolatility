# Predecting next-day stock volatility for risk management.
building a model that predicts next-day volatility for a stock using historical features to find the best features for predecting volatility.
This model uses MAE as the evaluation metric and GridSearchCV to calculate the score for each combination of parameters on the grid.
## To run the code
First you need to run data.py
This model uses Apple stock with ticker:"AAPL" in 2014-2024 as the data.
To replace the stock and date, in data.py, change `ticker = "AAPL"` and `df = yf.download(ticker, start="2014-01-01", end="2024-01-01")` with the desired stock and date.
then change `df.to_csv("AAPL_2014_224.csv")` with desired name to convert the downloaded file from pandas.DataFrame to .csv
now in main.py change `"AAPL_2014_2024.csv"` in `df = pd.read_csv("AAPL_2014_2024.csv", index_col=0, parse_dates=True)` to the name you chose.
Then run main.py
##Notes
GridSearchCV may take some time to find the best parameters as it fits 3 × 4 × 3 × 3 × 5 = 508 RandomForest Regressors. especially on CPU only and large datasets(in our case, 10 years)
With a Intel core i7-10510U CPU as the avrage, It should take around 140 seconds.
You can adjust `n_jobs` to control how many CPU cores are used, but `-1` uses all available cores.
You can use **XGBoost** library to run the model on GPU for acceleration.
