
import numpy as np
import pandas as pd
import sys
# Add parent directory to sys.path
# sys.path.append("..")
sys.path.append("/")
# Add grandparent directory to sys.path
sys.path.append("..")

from backtesting.machine_learning.optimization.optimizer import MlOptimizer
from backtesting.machine_learning.optimization.model_factory import ModelFactory
from utilities.plot_utils.ml_model_evaluator import MlDataPlotter
from utilities.performance import Performance


class MlBacktester:
    def __init__(self, config, data_manager):
        self.config = config
        self.scoring = None
        self.model = None
        self.model_params = None
        self.model_name = None
        # Use the provided arguments or set default values
        self.data_manager = data_manager
        self.data_plotter = None
        self.model_evaluator = None
        self.performance_evaluator = None
        self.results = {}
        self.optimize_model()

    def load_data(self):
        # Common implementation for loading data
        pass

    def set_data(self):
        # Common implementation for setting data
        pass

    def perform_backtest(self):
        # Common or placeholder implementation for backtesting
        pass

    def load_model(self):
        # Common implementation for loading a model
        pass

    def optimize_model(self):
        # Step 1: Instantiate the GridSearchOptimizer
        optimizer = MlOptimizer(self.x_train, self.y_train, self.config.symbol, task_type=self.config.task_type,
                                num_folds=self.config.num_folds, scoring=self.config.scoring)

        # Step 2: Use the optimizer's run_grid_search method to optimize the model
        param_grid = self.config.param_grid  # Assuming you have param_grid defined in your config
        best_model = optimizer.run_grid_search(self.config.model_name, self.model, param_grid)

        # Update the model of the strategy to the optimized model
        self.model = best_model

        # Get the optimized parameters
        optimized_params = self.model.get_params()
        print("Optimized Parameters:", optimized_params)

    def train_model(self):
        # Common implementation for training a model
        pass

    def evaluate_model(self):
        # Common implementation for evaluating a model
        pass

    def evaluate_backtest_results(self):
        # Common implementation for evaluating backtest results
        pass

    def _calculate_strategy_values(self,data_subset):
        '''Calculate strategy values.'''
        data_subset['strategy'] = data_subset["predictions"].shift() * data_subset['Returns']
        return data_subset

    def _calculate_trades(self, data_subset, y_pred_series):
        '''Determine when a trade takes place.'''

        data_subset["trades"] = y_pred_series.diff().ffill().fillna(0).abs()
        return data_subset

    def _adjust_for_transaction_costs(self, data_subset):
        '''Subtract transaction costs from strategy when a trade takes place.'''
        data_subset['strategy_net'] = data_subset['strategy'] - data_subset["trades"] * self.config.tc
        print(f"In _adjust_for_transaction_costs: {data_subset.columns}")
        return data_subset

    def _calculate_cumulative_metrics(self, data_subset):
        '''Calculate cumulative returns and strategy.'''
        data_subset['creturns'] = data_subset['Returns'].cumsum().apply(np.exp)
        data_subset['cstrategy'] = data_subset['strategy'].cumsum().apply(np.exp)
        data_subset['cstrategy_net'] = data_subset['strategy_net'].cumsum().apply(np.exp)

        return data_subset

    def _generate_predictions(self, data_subset):
        '''Generate predictions using the model.'''
        predictions = self.model.predict(data_subset[self.data_manager.feature_columns])
        data_subset['predictions'] = predictions
        data_subset['predictions'] = data_subset['predictions'].ffill().fillna(0)

        return data_subset

    # TODO: add the dataset_mode element to json: train, test, full
    def _select_data_based_on_mode(self, mode):
        '''Select data based on the provided mode ("train", "test" or "both")'''

        if mode == "train":
            return self.data_manager.X_train, self.data_manager.y_train
        elif mode == "test":
            return self.data_manager.X_test, self.data_manager.y_test
        elif mode == "full":
            # Create combined datasets for train and test data
            combined_X = pd.concat([self.data_manager.X_train, self.data_manager.X_test])
            combined_y = pd.concat([self.data_manager.y_train, self.data_manager.y_test])
            return combined_X, combined_y
        else:
            raise ValueError("Invalid mode. Choose from 'train', 'test', or 'both'.")

    def _filter_results_columns(self, data_subset, target_subset):
        '''Select only the desired columns for the results.'''
        desired_columns = ['creturns', 'cstrategy', 'cstrategy_net', 'strategy_net','Close', 'Volume', 'predictions', 'trades', 'strategy', 'Returns']
        data_subset = data_subset[desired_columns].copy()
        data_subset["Signal"] = target_subset
        return data_subset

    def _calculate_performance(self, data_subset):
        self.performance_evaluator = Performance(symbol=self.config.dataset_conf.symbol)
        self.performance_evaluator.set_data(data=data_subset)
        self.performance_evaluator.calculate_performance()

    def _plot_results(self, data_subset):

        self.plotter = MlDataPlotter(model_name=self.config.model_name,target=self.config.dataset_conf.target_conf.target, data_manager=self.data_manager)
        self.plotter.plot_performance_to_axis(config = self.config,
                                              data_subset=data_subset,
                                              perf_obj=self.performance_evaluator,
                                              **self.model_params)

        self.plotter.close_pdf()
        # TODO: Calculate performance Data

    def print_stats(self, data_subset):
        # Assuming you have a DataFrame named data_subset
        strategy = data_subset['strategy']

        max_val = strategy.max()
        mean_val = strategy.mean()
        std_val = strategy.std()

        print(f"Statistics for 'strategy' series:")
        print(f"Len: {len(strategy)}")
        print(f"Max: {max_val}")
        print(f"Mean: {mean_val}")
        print(f"Standard Deviation: {std_val}")

        returns = data_subset['Returns']
        max_val = returns.max()
        mean_val = returns.mean()
        std_val = returns.std()
        print("=" * 120)
        print(f"Statistics for 'returns' series:")
        print(f"Len: {len(returns)}")
        print(f"Max: {max_val}")
        print(f"Mean: {mean_val}")
        print(f"Standard Deviation: {std_val}")

        predictions = data_subset['predictions']

        # Calculate the number and proportion of 1s and 0s
        total_ones = predictions.sum()  # Summing will count the number of 1s
        total_zeros = len(predictions) - total_ones

        proportion_ones = total_ones / len(predictions)
        proportion_zeros = total_zeros / len(predictions)

        print(f"Statistics for 'prediction' series:")
        print(f"Total Observations: {len(predictions)}")
        print(f"Number of 1s: {total_ones} (Proportion: {proportion_ones:.4f})")
        print(f"Number of 0s: {total_zeros} (Proportion: {proportion_zeros:.4f})")

    def run_strategy(self):
        '''Backtests the trading strategy.'''

        data_subset, target_subset = self._select_data_based_on_mode(self.config.dataset_conf.mode)
        data_subset = self._generate_predictions(data_subset)

        # Calculate strategy values
        data_subset = self._calculate_strategy_values(data_subset)
        self.print_stats(data_subset)
        # Make model predictions
        # Identify the trades
        data_subset = self._calculate_trades(data_subset, target_subset)

        # Adjust for transaction costs
        data_subset = self._adjust_for_transaction_costs(data_subset)

        # Calculate cumulative metrics
        data_subset = self._calculate_cumulative_metrics(data_subset)
        # Use the method to filter desired columns
        data_subset = self._filter_results_columns(data_subset, target_subset)
        print("after filter columns")
        print(data_subset.head())
        # Store results for the current mode (train, test, or both)
        data_subset.dropna(inplace=True)
        self.data_manager.data_subset = data_subset
        self._calculate_performance(data_subset)

        self._plot_results(data_subset)
        exit(0)

    def get_model_params(self):
        model_dict = {model_config.model: model_config.params for model_config in self.config.models_config}
        params = model_dict.get(self.config.model_name)
        return params

    def _validate_model(self):
        self.data_manager.y_pred_train = self.model.predict(self.data_manager.X_train[self.data_manager.feature_columns])
        self.data_manager.y_pred_test = self.model.predict(self.data_manager.X_test[self.data_manager.feature_columns])

    def optimize_model(self):
        # Step 1: Instantiate the GridSearchOptimizer
        model = ModelFactory.get_model(self.config.model_name, self.config.model_type)
        # TODO: update the config to consider the num_folds, scoring, ..-> add model_config element
        optimizer = MlOptimizer(self.data_manager.X_train[self.data_manager.feature_columns], self.data_manager.y_train,
                                self.config.dataset_conf.symbol,
                                task_type=self.config.model_type)

        # Step 2: Use the optimizer's run_grid_search method to optimize the model
        param_grid = self.get_model_params()
        best_model = optimizer.run_grid_search(self.config.model_name, model(), param_grid)
        # Update the model of the strategy to the optimized model
        self.model = best_model

        # Store the optimized parameters as an attribute
        self.model_params = self.model.get_params()
        print("="*120)
        print("\n")
        print(f"Parameters:")
        print(self.model_params)
        print("=" * 120)
        print("\n")
        self._validate_model()


