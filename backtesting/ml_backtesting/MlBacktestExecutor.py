import sys
# Add parent directory to sys.path
from pydantic import ValidationError
import json
sys.path.append("..")
# Add grandparent directory to sys.path
sys.path.append("../..")

config_file = "./backtester_config.json"
from backtester_config import BacktesterConfig
from backtesting.machine_learning.MlBacktester import MlBacktester

from utilities.data_utils.ml_data_manager import MlDataManager
from utilities.plot_utils.ml_model_evaluator import MlDataPlotter



class MlBacktestExecutor:
    def __init__(self):
        self.config = None

    def load_config(self):
        try:
            # Validate the path
            if not config_file.endswith('.json'):
                raise ValueError(f"Invalid path. Expected a .json file, got {config_file}")

            # Read the file
            with open(config_file, 'r') as file:
                data = json.load(file)

            # Parse the data using the Pydantic model
            self.config = BacktesterConfig(**data)

        except FileNotFoundError:
            print(f"Error: The file {config_file} was not found.")
        except json.JSONDecodeError:
            print("Error: The provided JSON file has invalid syntax.")
        except ValidationError as e:
            print("Error: Invalid configuration format.")
            print(e)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    backtester = MlBacktestExecutor()
    backtester.load_config()

    # Define the list of targets based on your information
    targets = ['Simple', 'MA_Relative', 'Momentum', 'ROC', 'RSI_Cross', 'Consecutive_Increases', 'Exceed_Threshold',
               'Volatility_Breakout', 'Exceed_Avg_Returns']
    targets_ = ['Simple', 'MA_Relative']
    symbols = ["XRPUSDT", "BTCUSDT", "ETHUSDT", "LTCUSDT", "TRXUSDT"]

    #for symbol in symbols:
    #backtester.config.dataset_conf.symbol = symbol
    #for target in targets:

    #backtester.config.dataset_conf.target_conf.target = target

    data_manager = MlDataManager(dataset_conf=backtester.config.dataset_conf)
    data_manager.preprocess_data()

    strategy = MlBacktester(backtester.config, data_manager=data_manager)
    strategy.run_strategy()

    data_plotter = MlDataPlotter(data_manager=data_manager, target=backtester.config.dataset_conf.target_conf.target, model_name=backtester.config.model_name)

    columns_to_remove = ['Open', 'High', 'Low', 'Signal', 'DOW', 'Roll_Rets', 'Roll_Rets']

    # Also remove columns that contain '10' in their names
    columns_to_remove.extend([col for col in data_manager.data.columns if '30' in col or '70' in col])

    data_manager.data = data_manager.data.drop(columns=columns_to_remove)

    # Plot various metrics and visualizations
    #data_plotter.plot_signal_distribution()
    # data_plotter.plot_covariance_matrix()
    #data_plotter.plot_classification_report(train=True)
    #data_plotter.plot_classification_report(train=False)
    # data_plotter.plot_model_evaluation()
    # data_plotter.plot_confusion_matrix_values()
    #data_plotter.plot_feature_importance(strategy.model)
    # data_plotter.plot_performance_to_axis(config=backtester.config,
    #                                       data_subset=strategy.data_manager.data_subset,
    #                                       perf_obj=strategy.performance_evaluator,
    #                                       **strategy.model_params)
    #data_plotter.plot_explained_variance()
    #data_plotter.plot_tsne()

    data_plotter.close_pdf()
