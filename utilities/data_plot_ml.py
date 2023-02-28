
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base_str_data = ['ticker', 'model_name', 'accuracy_train', 'accuracy_val', 'precision', 'recall',
                 'f1', 'TN', 'FP', 'FN', 'TP', 'freq', 'train_samples', 'val_samples']


class DataPlotML:

    def __init__(self,  symbol):
        self.symbol = symbol
        self.init_df()

    def init_df(self):
        self.df = pd.DataFrame(columns=base_str_data)
        #self.df.set_index('performance', inplace=True)

    def store_testcase_data(self, perf_obj):
        testcase_data = []
        testcase_data.append(perf_obj.symbol)
        testcase_data.append(perf_obj.model_name)
        testcase_data.append(perf_obj.perf_metrics_dic['accuracy_train'])
        testcase_data.append(perf_obj.perf_metrics_dic['accuracy_val'])
        testcase_data.append(perf_obj.perf_metrics_dic['precision'])
        testcase_data.append(perf_obj.perf_metrics_dic['recall'])
        testcase_data.append(perf_obj.perf_metrics_dic['f1'])
        testcase_data.append(perf_obj.perf_metrics_dic['TN'])
        testcase_data.append(perf_obj.perf_metrics_dic['FP'])
        testcase_data.append(perf_obj.perf_metrics_dic['FN'])
        testcase_data.append(perf_obj.perf_metrics_dic['TP'])
        testcase_data.append(perf_obj.freq)
        testcase_data.append(len(perf_obj.X_train))
        testcase_data.append(len(perf_obj.X_val))

        self.df.loc[len(self.df)] = testcase_data

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