from MlDataPreparer import MlDataPreparer


class MlBacktesterBase(MlDataPreparer):

    def __init__(self, filepath, symbol, ml=True, tc=0.00007, start=None, end=None, dataset="training"):
        self.filepath = filepath
        self.symbol = symbol
        self.start = start
        self.end = end
        self.tc = tc
        self.results = None
        self.dataset = dataset
        self.test_size = 0.2
        self.get_data(ml=ml)
        self.results_folder = "../results/ml"
        self.tp_year = (self.data.Close.count() / ((self.data.index[-1] - self.data.index[0]).days / 365.25))

    def select_data(self, start, end):
        ''' Selects sub-sets of the financial data. '''
        self.data = self.data[(self.data.index >= start) & (self.data.index <= end)].copy()

    def run_backtest(self):
        ''' Runs the strategy backtest.
        '''
        data = self.results.copy()
        data["strategy"] = data["position"].shift(1) * data["returns"]

        # determine the number of trades in each bar
        data["trades"] = data.position.diff().fillna(0).abs()

        # subtract transaction/trading costs from pre-cost return
        data.strategy = data.strategy - data.trades * self.tc

        self.results = data

    def upsample(self):
        '''  Upsamples/copies trading positions back to higher frequency.
        '''

        data = self.data.copy()
        resamp = self.results.copy()

        data["position"] = resamp.position.shift()
        data = data.loc[resamp.index[0]:].copy()
        data.position = data.position.shift(-1).ffill()
        data.dropna(inplace=True)
        self.results = data