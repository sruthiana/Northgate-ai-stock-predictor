# Northgate AI Stock Predictor
## Final Project Report

### 1. Abstract

Northgate AI Stock Predictor is an educational financial analytics project developed to explore the integration of machine learning, deep learning, news sentiment, portfolio optimization, and historical backtesting.

The workflow includes historical market data processing, technical and macroeconomic feature engineering, traditional machine learning evaluation, LSTM and GRU modeling, trading signal generation, portfolio optimization, and out-of-sample strategy evaluation.

The project also integrates optimized portfolio backtests and prediction-based strategies into a unified performance dataset and visualization.

### 2. Problem Statement

Financial market data contains complex patterns influenced by historical price movements, volatility, macroeconomic conditions, and market sentiment.

The project investigates whether a structured machine learning workflow can produce useful return predictions and how portfolio allocation methods perform when evaluated against a market benchmark.

The objective is analytical and educational rather than the development of a guaranteed profitable trading system.

### 3. Objectives

1. Build a historical market data processing pipeline.
2. Generate technical and macroeconomic features.
3. Incorporate available news sentiment.
4. Evaluate traditional machine learning models.
5. Explore recurrent neural networks for return prediction.
6. Generate trading signals from predictions.
7. Optimize portfolio weights.
8. Evaluate portfolios through historical backtesting.
9. Compare strategies using risk and return metrics.
10. Integrate outputs into a unified portfolio comparison.

### 4. Data and Features

The configured stock universe contains ten equities:

AAPL, MSFT, JPM, XOM, JNJ, PG, NVDA, KO, CAT, and HD.

The project also uses the S&P 500, the VIX, and selected macroeconomic variables.

Feature engineering includes:
- Daily and logarithmic returns
- Moving averages
- Exponential moving averages
- RSI
- MACD
- Bollinger Bands
- Rolling volatility
- Treasury yield spread
- CPI-related features
- Unemployment changes
- News sentiment counts

The processed modeling dataset contains 31 numeric features before final complete-feature filtering.

### 5. Modeling Workflow

Traditional machine learning models were evaluated alongside recurrent deep learning models.

The deep learning experiments included LSTM and GRU architectures using historical sequences.

The target was the next-day logarithmic return.

Chronological data splitting was used to reduce temporal leakage.

The project evaluated predictive errors and directional performance rather than relying solely on strategy returns.

### 6. Model Evaluation

The traditional model evaluation identified limitations in predicting next-day returns.

The deep learning evaluation produced the following test results for LSTM:

- MAE: approximately 0.012125
- RMSE: approximately 0.017509
- R²: approximately -0.012262
- Directional accuracy: approximately 50.87%

The zero-return baseline test RMSE was approximately 0.017417.

The LSTM RMSE was slightly worse than the zero-return baseline. Therefore, the model should not be described as demonstrating superior next-day return prediction.

### 7. Portfolio Optimization

The project evaluated three portfolio allocation approaches:

- Equal Weight
- Minimum Volatility
- Maximum Sharpe

Portfolio optimization was followed by out-of-sample backtesting.

The backtest used a training period for portfolio estimation and a subsequent test period for evaluation.

### 8. Portfolio Backtest Results

| Strategy | Cumulative Return | Annualized Return | Volatility | Sharpe | Maximum Drawdown |
|---|---:|---:|---:|---:|---:|
| Maximum Sharpe | 136.16% | 35.66% | 23.32% | 1.36 | -25.78% |
| Equal Weight | 97.56% | 27.34% | 12.58% | 1.85 | -15.71% |
| Minimum Volatility | 60.11% | 18.18% | 12.89% | 1.10 | -10.71% |
| S&P 500 | 58.76% | 17.80% | 15.14% | 0.91 | -18.90% |

Maximum Sharpe produced the highest cumulative return in this backtest but also the largest maximum drawdown.

Equal Weight produced the highest reported Sharpe ratio among these strategies.

Minimum Volatility produced the smallest maximum drawdown.

These are historical results, not forecasts.

### 9. LSTM Strategy Results

The prediction-based strategy comparison produced:

| Strategy | Cumulative Return | Annualized Return | Volatility | Sharpe | Maximum Drawdown |
|---|---:|---:|---:|---:|---:|
| LSTM Long Only | 12.85% | 5.84% | 8.33% | 0.22 | -9.02% |
| LSTM Long Short | 3.07% | 1.43% | 8.85% | -0.29 | -8.78% |

The LSTM Long Only strategy produced a positive historical return.

The LSTM Long Short strategy produced a lower cumulative return and a negative Sharpe ratio.

The strategy results should be considered exploratory because predictive performance did not demonstrate an improvement over the zero-return baseline on test RMSE.

### 10. Unified Integration

The integration module successfully loaded the optimized portfolio backtest and prediction-based strategy comparison.

The resulting unified comparison contains six strategies:

1. Maximum Sharpe Optimized Backtest
2. Equal Weight Optimized Backtest
3. Minimum Volatility Optimized Backtest
4. S&P 500 Optimized Backtest Benchmark
5. LSTM Long Only
6. LSTM Long Short

The integration generated:
- Unified comparison CSV
- Unified daily equity CSV
- Unified performance chart

The unified daily dataset contains 3,915 rows and four columns:
Date, Strategy, Equity, and Daily_Return.

### 11. Final Validation

Structural validation confirmed:
- No duplicate dates within individual strategy curves
- No non-positive equity values
- No missing values in the unified daily dataset

The initial reconciliation formula was corrected to use normalized initial capital.

Final equity return was calculated as:

Final normalized equity minus one.

After correction, compounded daily returns matched equity-based returns for all six strategies.

The final validation therefore confirmed consistency between the daily return series and the normalized equity curves.

### 12. Discussion

The project demonstrates a complete educational financial analytics pipeline.

The portfolio optimization results were stronger than the prediction-based strategies in the historical evaluations, but the comparison periods differ.

The Maximum Sharpe portfolio showed strong historical returns accompanied by substantial drawdown.

Equal Weight provided a more moderate risk-return profile in the reported backtest.

The LSTM strategies generated positive cumulative returns, but the model's predictive RMSE did not beat the zero-return baseline.

News sentiment was available but sparse, limiting the strength of conclusions about sentiment-driven predictive improvement.

These findings highlight the importance of separating predictive accuracy from portfolio performance and of evaluating financial strategies using consistent periods, realistic costs, and robust out-of-sample methods.

### 13. Limitations

1. Historical returns cannot guarantee future performance.
2. Financial markets are non-stationary.
3. News sentiment observations are sparse.
4. LSTM predictive RMSE did not outperform the zero-return baseline.
5. Portfolio and LSTM strategy comparisons use different periods.
6. Results depend on data availability, preprocessing, transaction costs, and modeling assumptions.
7. Backtests may not capture all real-world execution effects.
8. This project is not a live trading or investment recommendation system.

### 14. Conclusion

Northgate AI Stock Predictor successfully implemented an educational workflow integrating market data processing, feature engineering, traditional machine learning, recurrent neural networks, news sentiment, trading signals, portfolio optimization, backtesting, and unified performance reporting.

The portfolio integration and final equity reconciliation completed successfully.

The project also identified important limitations in next-day return prediction and demonstrated the need for careful out-of-sample evaluation, risk analysis, and transparent reporting.

### 15. Future Work

Potential improvements include:
- More robust sentiment coverage
- Additional market and fundamental features
- Walk-forward validation
- Consistent evaluation windows across strategies
- Improved transaction cost and execution modeling
- Portfolio constraints and turnover controls
- Benchmark-aware risk analysis
- Further model calibration and baseline comparison

### Disclaimer

This report describes an educational and research prototype. It does not provide financial advice or investment recommendations.
