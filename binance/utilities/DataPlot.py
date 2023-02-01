import os.path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base_str_data = ['ticker', 'indicator', 'dataset',  'num_trades', 'strat_multiple',
                 'bh_multiple', 'outperf_net', 'outperf', 'cagr', 'ann_mean', 'ann_std', 'sharpe', 'sortino',
                 'max_dd', 'calmar', 'max_dd_dur', 'kelly_crit', 'start_d',
                 'end_d', 'num_samples']


class DataPlot:

    def __init__(self, dataset, indicator, symbol):
        self.dataset = dataset
        self.indicator = indicator
        self.symbol = symbol
        self.init_df()

    def init_df(self):
        self.df = pd.DataFrame(columns=base_str_data)
        #self.df.set_index('performance', inplace=True)

    def store_testcase_data(self, perf_obj, params):
        testcase_data = []
        testcase_data.append(self.symbol)
        testcase_data.append(self.indicator)
        testcase_data.append(self.dataset)
        testcase_data.append(perf_obj.num_of_trades)
        testcase_data.append(perf_obj.strategy_multiple)
        testcase_data.append(perf_obj.bh_multiple)
        testcase_data.append(perf_obj.outperf_net)
        testcase_data.append(perf_obj.outperf)
        testcase_data.append(perf_obj.cagr)
        testcase_data.append(perf_obj.ann_mean)
        testcase_data.append(perf_obj.ann_std)
        testcase_data.append(perf_obj.sharpe)
        testcase_data.append(perf_obj.sortino)
        testcase_data.append(perf_obj.max_drawdown)
        testcase_data.append(perf_obj.calmar)
        testcase_data.append(perf_obj.max_dd_duration)
        testcase_data.append(perf_obj.kelly_criterion)
        testcase_data.append(perf_obj.start_d)
        testcase_data.append(perf_obj.end_d)
        testcase_data.append(perf_obj.num_samples)

        for key, value in params.items():
            if key not in self.df.columns:
                #base_str_data.append(key)
                self.df[key] = np.nan
            testcase_data.append(value)

            #self.df[key] = value
            #print(key, value)
        self.df.loc[len(self.df)] = testcase_data

    def store_data(self, output_dir):
        path = os.path.join(output_dir, f"{self.symbol}_{self.indicator}.csv")
        self.df.to_csv(path)
        print(f"INFOR: Stored the test data into {path}..")
    def create_col(self):
        lst = []

        ticker = "XRP"
        strategy = "EMA"
        dataset = "training"
        freq = "40min"
        num_trades = 236
        stat_mul = 0.95

        lst.append(ticker)
        lst.append(strategy)
        lst.append(dataset)
        lst.append(freq)
        lst.append(num_trades)
        lst.append(stat_mul)

        return lst
