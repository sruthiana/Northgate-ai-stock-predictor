# Northgate AI Stock Predictor

## 1. Project Overview

Northgate AI Stock Predictor is an educational machine learning and portfolio analytics project that combines historical stock market data, technical indicators, macroeconomic variables, news sentiment, predictive modeling, trading signals, portfolio optimization, and backtesting.

The project explores how machine learning predictions and portfolio allocation strategies can be evaluated using historical market data.

This project is for educational and research purposes only. It is not financial advice or a live trading system.

## 2. Objectives

- Collect and process historical stock market data.
- Engineer technical and macroeconomic features.
- Integrate news sentiment information.
- Train and evaluate traditional machine learning models.
- Explore LSTM and GRU time-series prediction.
- Generate prediction-based trading signals.
- Optimize portfolios using different allocation strategies.
- Perform out-of-sample portfolio backtesting.
- Compare strategy returns, volatility, Sharpe ratio, and drawdown.
- Produce reports, charts, and unified performance outputs.

## 3. Technology Stack

- Python
- pandas
- NumPy
- scikit-learn
- TensorFlow / Keras
- PyTorch
- XGBoost
- PyPortfolioOpt
- CVXPY
- Matplotlib
- Plotly
- YAML configuration

## 4. Market Data

The project universe includes:

AAPL, MSFT, JPM, XOM, JNJ, PG, NVDA, KO, CAT, and HD.

Market reference:
- S&P 500: ^GSPC
- Volatility index: ^VIX

Macroeconomic variables include:
- Treasury yields
- CPI
- Unemployment rate

The configured historical period begins in 2015 and extends through October 2026, subject to data-source availability and the configured end-date convention.

## 5. Modeling

The project evaluates traditional machine learning models and deep learning approaches.

Traditional models include:
- Linear Regression
- Random Forest

Deep learning approaches include:
- LSTM
- GRU

The prediction target is the next-day log return.

The project uses chronological training, validation, and testing periods to evaluate predictive performance.

## 6. News Sentiment

News sentiment is incorporated into the processed datasets and strategy analysis.

Sentiment features include:
- News count
- Positive count
- Neutral count
- Negative count

Sentiment coverage is sparse in the available dataset. The sentiment contribution should therefore be interpreted cautiously.

## 7. Portfolio Optimization

Portfolio strategies include:

- Equal Weight
- Minimum Volatility
- Maximum Sharpe

The project evaluates portfolio allocation using historical returns and compares optimized portfolios against the S&P 500 benchmark.

## 8. Backtesting

The project includes out-of-sample portfolio backtesting and prediction-based strategy comparison.

Reported metrics include:
- Cumulative return
- Annualized return
- Annualized volatility
- Sharpe ratio
- Maximum drawdown
- Win rate

The optimized portfolio backtest and LSTM strategy comparison have different evaluation periods. Their returns should not be interpreted as a perfectly controlled comparison.

## 9. Unified Portfolio Integration

The integration module combines the existing portfolio backtest and LSTM strategy comparison outputs.

Run:

```powershell
python -m src.ai_portfolio_integration
```

Generated outputs:

- reports/ai_portfolio_unified_comparison.csv
- reports/ai_portfolio_unified_daily.csv
- reports/figures/ai_portfolio_unified_performance.png

## 10. Final Validation

The unified daily dataset contains 3,915 rows across six strategies.

Structural validation confirmed:
- No duplicate dates within strategies
- No non-positive equity values
- No missing values in the unified daily dataset

Equity reconciliation confirmed that compounded daily returns match final normalized equity minus one for all six strategies.

## 11. Important Results

Out-of-sample optimized portfolio backtest:

| Strategy | Cumulative Return | Annualized Return | Sharpe | Maximum Drawdown |
|---|---:|---:|---:|---:|
| Maximum Sharpe | 136.16% | 35.66% | 1.36 | -25.78% |
| Equal Weight | 97.56% | 27.34% | 1.85 | -15.71% |
| Minimum Volatility | 60.11% | 18.18% | 1.10 | -10.71% |
| S&P 500 | 58.76% | 17.80% | 0.91 | -18.90% |

Prediction-based strategy comparison:

| Strategy | Cumulative Return | Annualized Return |
|---|---:|---:|
| LSTM Long Only | 12.85% | 5.84% |
| LSTM Long Short | 3.07% | 1.43% |

These results are historical backtest outputs and do not guarantee future performance.

## 12. Project Structure

```text
northgate_stock_ai_phase1/
├── config/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
├── reports/
│   └── figures/
├── src/
├── tests/
├── requirements.txt
└── README.md
```

Actual folders and files may vary according to the project implementation.

## 13. Environment Setup

Create and activate a Python virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 14. Execution

Run the project modules according to their configured workflow.

The unified portfolio integration can be executed with:

```powershell
python -m src.ai_portfolio_integration
```

The project requires the corresponding processed data and prior model/backtest outputs.

## 15. Limitations

- Historical predictive performance does not guarantee future results.
- Financial markets are subject to regime changes and unpredictable events.
- News sentiment coverage is sparse.
- LSTM test performance did not outperform the zero-return baseline on RMSE.
- Portfolio strategies have different evaluation periods.
- Transaction costs, data quality, survivorship, and model assumptions affect results.
- The project is an educational prototype and is not intended for investment decisions.

## 16. Conclusion

Northgate AI Stock Predictor demonstrates an end-to-end educational workflow covering data processing, feature engineering, machine learning, deep learning, sentiment analysis, trading signals, portfolio optimization, backtesting, and performance reporting.

The unified portfolio integration and final equity reconciliation have been completed successfully.

The project provides a foundation for further research into time-series modeling, sentiment integration, risk-aware allocation, and robust financial evaluation.

---

**Disclaimer:** This project is for educational and research purposes only. It does not provide financial advice, investment recommendations, or guarantees of future returns.
