# Backtesting with Optimization

This project implements backtesting strategies for trading using technical indicators, such as MACD, EMA, RSI, BB, SO, and RSI. Data from Binance is used for backtesting, which is downloaded using the DataRetriever module.

## Features
- Implements backtesting strategies using technical indicators and Machine Learning algorithms. Currently, only Logistic Regression is implemented for generating buy or sell signals. But the framework is flexible, other ML or DL algorithms can be added, trained and used for backtesting. 
- Supports iterative and vectorized backtesting.
- Optimizes parameters for technical indicators within a configurable parameter range.
- Supports individual configuration for the optimization of each technical indicator.
- Downloads data from Binance using the DataRetriever module.
- Stores the results of backtests and optimization for further analysis.
- Supports logging and email sending with notifications about backtesting results and trading sessions in case of deployment.
- The "Client" module handles deployment of trained and optimized modules. It includes a client class to connect to Binance and execute trades.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
- Python 3.6 or higher
- pip package manager

### Installation
- Clone the repository to your local machine and navigate to the project directory and
- Install the required packages: `pip install -r requirements.txt`

## Configuration and Execution
You can configure the backtesting by modifying the JSON files in the `config` folder under `vectorized_backtesting`. The `strategy_config.json` file allows you to configure the parameters for the technical indicators, as well as the frequency of the time series data. An example configuration for the EMA technical indicator looks like this:

```json
{
    "freq" : [5, 720, 5],
    "metric": "Multiple",
    "ema": {
        "ema_s": [10, 50, 5],
        "ema_l": [50, 150, 10]
    }
}
```

In this example, a grid search is performed to find the best combination of parameters for the EMA and the optimal time interval for the time series data. The "Multiple" metric is used to evaluate the performance of the parameters, although other metrics such as "Calmar", "Sharpe"  or "Sortino" can be used, if you want to consider risk in the backtesting.

The `backtester_config.json` file allows you to configure various aspects of the backtesting, such as the start and end times for historical data, training and testing sample size, tickers, etc.

# Usage
After configuration of the backtesting using the JSON files, you can start the project. As an example, let's use the vectorized backtester. Navigate to the `vectorized_backtesting` folder and run the `python backtester.py` in terminal.

The vectorized backtester will then use the configuration provided to grid search the parameters for the configured technical indicators. This will start the backtesting process, storing the results as CSV files in the `results` directory. These results can be analyzed using the `DataPlotter` module.

## Deployment

The `Client` module is responsible for deploying the trained and optimized modules. It includes a client class that connects to Binance and executes trades. This module is designed to enable seamless execution of trades based on the output from your backtested strategies.

To setup your Binance client, please provide the necessary API credentials in the `credentials.py` file located in the `utilities` folder. You will need your own API keys from Binance for this purpose.

Please ensure to keep your API keys secure and do not commit them into version control.

## Supported Brokers and Markets

Despite the fact that only cryptocurrency data was used and Binance has been integrated for live trading, the backtester can be utilized for backtesing for a variety of markets beyond cryptocurrencies. It can be applied to any market data, including stocks, commodities, forex, etc., provided the data is in a structured CSV format.

The data must adhere to the following structure: `date, open, high, low, close, volume`. This ensures compatibility with the technical indicators and other functions within the backtesting framework.

This flexibility allows you to perform historical simulation and analysis on a broad range of market data to refine and optimize your strategies.
