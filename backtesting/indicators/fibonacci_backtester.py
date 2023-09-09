def prepare_data(self, freq, fib_levels):
    ''' Prepares the Data for Backtesting.
    '''
    freq = f"{freq}min"
    data_resampled = self.data.resample(freq).last().dropna().iloc[:-1].copy()
    data_resampled["returns"] = np.log(1 + data_resampled.Close.pct_change())
    data_resampled.dropna()

    ######### INSERT THE STRATEGY SPECIFIC CODE HERE ##################
    high_price = data_resampled['High']
    low_price = data_resampled['Low']
    close_price = data_resampled['Close']
    highest_high = high_price.rolling(window=fib_levels, min_periods=fib_levels).max()
    lowest_low = low_price.rolling(window=fib_levels, min_periods=fib_levels).min()
    diff = highest_high - lowest_low
    levels = [0.0] + [0.236, 0.382, 0.5, 0.618, 0.786, 1.0]  # list of fibonacci levels
    fib_levels = pd.Series(levels, index=levels) * diff[-1] + lowest_low[-1]
    ###################################################################

    position = np.zeros(len(data_resampled))
    # Your position calculation logic goes here
    position = pd.Series(position, index=data_resampled.index).ffill().fillna(0)
    data_resampled = data_resampled.assign(position=position)
    self.results = data_resampled

def optimize_strategy(self, freq_range, fib_levels_range, metric="Multiple"):
    '''
    Backtests strategy for different parameter values incl. Optimization and Reporting.

    Parameters
    ============
    freq_range: tuple
        A tuple of the form (start, end, step size) specifying the range of frequencies to be tested.

    fib_levels_range: tuple
        A tuple of the form (start, end, step size) specifying the range of Fibonacci levels to be tested.

    metric: str (default: "Multiple")
        A performance metric to be optimized, which can be one of the following: "Multiple", "Sharpe", "Sortino", "Calmar", or "Kelly".
    '''

    # Use only OHLC prices
    self.data = self.data.loc[:, ["Open", "High", "Low", "Close"]]
    # performance_function = self.perf_obj.performance_functions[metric]

    freqs = range(*freq_range)
    fib_levels = range(*fib_levels_range)

    combinations = list(product(freqs, fib_levels))
    # remove combinations there freq > 180 and not multiple of 30
    combinations = list(filter(lambda x: x[0] <= 180 or x[0] % 30 == 0, combinations))

    for (freq, fib_levels_val) in tqdm(combinations):
        self.generate_signals(freq, fib_levels_val)
        if metric != "Multiple":
            self.upsample()
        self.run_backtest()
        # set strategy data and calculate performance
        self.perf_obj.set_data(self.results)
        self.perf_obj.calculate_performance()
        # store strategy performance data for further plotting
        params_dic = {"freq": freq, "fib_levels": fib_levels_val}
        self.dataploter.store_testcase_data(self.perf_obj, params_dic) # comb[0] is current data freq

    logger.info(f"Total number of executed tests: {len(combinations)}.")


def prepare_data(self, freq, level, lookback_period):
    ''' Prepares the Data for Backtesting.
    '''
    freq = f"{freq}min"
    data_resampled = self.data.resample(freq).last().dropna().iloc[:-1].copy()
    data_resampled["returns"] = np.log(1 + data_resampled.Close.pct_change())
    data_resampled.dropna()

    # Compute the Fibonacci retracements
    high = data_resampled.High.rolling(window=lookback_period).max()
    low = data_resampled.Low.rolling(window=lookback_period).min()
    diff = high - low
    levels = high - diff * level / 100.0

    # Take long positions when the price is above a certain Fibonacci level
    position = np.zeros(len(data_resampled))
    for i in range(1, len(data_resampled)):
        if data_resampled.Close[i] > levels[i]:
            position[i] = 1

    position = pd.Series(position, index=data_resampled.index).ffill().fillna(0)
    data_resampled = data_resampled.assign(position=position)
    self.results = data_resampled
