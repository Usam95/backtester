# Backtesting with optimization

This project implements backtesting strategies for trading using technical indicators, such as MACD, EMA, RSI, BB, SO, and RSI. The project uses data from Binance, which is downloaded using the `DataRetriever` module.

 
## Features

- Implements backtesting strategies using Technical Indicators, such as MACD, EMA, RSI, BB, SO, and RSI.
- Optimizes the parameters for Technical Indicators using the configured parameters range
- Supports individual configuration for optimization of  each Technical Indicators
- Downloads data from Binance using the DataRetriever module.
- Stores the results of backtests and optimization for further analysing.
- Supports logging and email sending with notifications about backtesting results and trading sessions in case of deployment

Deployment of trained and optimized modules is handled by the "Client" module, which includes a client class to connect to Binance and execute trades.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

###  Prerequisites

- Python 3.6 or higher

- pip package manager

### Installation

Clone the repository to your local machine and navigate to the project directory and

Install the required packages:

- pip install -r requirements.txt

 
## Usage

To use the project, run the following command in the terminal:

`python main.py`

- This will start the backtesting process using the configuration defined in the JSON files. The results will be stored as CSV files in the results directory, which can be analyzed using the DataPlotter module.

 

Deployment

The "Client" module is responsible for deploying the trained and optimized modules. It includes a client class that connects to Binance and executes trades.

 

Contributing

If you'd like to contribute to this project, please follow these guidelines:

 

Fork the repository

Create a new branch for your feature (git checkout -b my-new-feature)

Commit your changes (git commit -am 'Add some feature')

Push to the branch (git push origin my-new-feature)

Create a new Pull Request
