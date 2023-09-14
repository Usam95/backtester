from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any
import os
from .model_factory import ModelFactory
from sklearn.datasets import load_iris
from .opt_config.models_config import AllModelsConfig
import json
from sklearn.linear_model import LogisticRegression, LinearRegression
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from typing import Dict, Any
import numpy as np
from sklearn.model_selection import StratifiedKFold
import joblib




class GridSearchOptimizer:
    def __init__(self, x_train, y_train, symbol, task_type="classification", num_folds=3, scoring=None, verbose=True):
        self.x_train = x_train
        self.y_train = y_train
        self.num_folds = num_folds
        self.verbose = verbose
        self.symbol = symbol
        if not scoring:
            self.scoring = 'accuracy' if task_type == "classification" else 'neg_mean_squared_error'
        else:
            self.scoring = scoring

        self.best_scores = {}  # to store the best score of each model
        self.best_params = {}  # to store the best parameters of each model
        self.all_results = {}  # to store all results of each model

    def _check_and_convert_parameters(self, model_name: str, param_grid: Dict[str, Any]) -> Dict[str, Any]:
        models_with_n_estimators = [
            "AdaBoostClassifier",
            "GradientBoostingClassifier",
            "RandomForestClassifier",
            "ExtraTreesClassifier"
        ]

        if model_name in models_with_n_estimators and "n_estimators" in param_grid:
            param_grid["n_estimators"] = [int(n) for n in param_grid["n_estimators"]]

        if model_name == "KNeighborsClassifier" and "n_neighbors" in param_grid:
            param_grid["n_neighbors"] = [int(n) for n in param_grid["n_neighbors"]]

        return param_grid

    def print_results(self):
        """
        Print the stored results of the grid search.
        """
        for model_name, results in self.all_results.items():
            print(f"Model: {model_name}")
            print("===================================")

            # Print best score and parameters
            best_score = self.best_scores.get(model_name, None)
            best_params = self.best_params.get(model_name, None)
            if best_score is not None and best_params is not None:
                print(f"Best Score: {best_score}")
                print(f"Best Parameters: {best_params}")
                print("-----------------------------------")

            # Print all results
            means = results['means']
            stds = results['stds']
            params = results['params']
            ranked_indices = results['ranked_indices']

            for idx in ranked_indices:
                mean = means[idx]
                stdev = stds[idx]
                param = params[idx]
                rank = idx + 1  # Since index starts at 0
                print(f"Rank: #{rank}")
                print(f"Mean Score: {mean}")
                print(f"Standard Deviation: {stdev}")
                print(f"Parameters: {param}")
                print("-----------------------------------")

            print("\n")  # Add a newline for separation between models

    def _save_model(self, model_name, trained_model, performance_info):
        """
        Save the trained model and its performance info.
        """
        # Define directory to save the model
        dir_path = f'./trained_models/{model_name}/{self.symbol}'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Save the trained model
        joblib.dump(trained_model, f'{dir_path}/trained_model.pkl')

        # Save the performance info in a text file
        with open(f'{dir_path}/performance_info.txt', 'w') as file:
            file.write(performance_info)

    def run_grid_search(self, model_name: str, model, param_grid: Dict[str, Any], rescale=True):
        print(f"Running Grid Search for {model_name}...")

        # Check and convert n_estimators if necessary
        param_grid = self._check_and_convert_parameters(model_name, param_grid)

        # Optionally rescale data
        if rescale:
            scaler = StandardScaler().fit(self.x_train)
            processed_x = scaler.transform(self.x_train)
        else:
            processed_x = self.x_train
        # Define time series split
        tscv = TimeSeriesSplit(n_splits=self.num_folds)
        #strat_kfold = KFold(n_splits, shuffle=True, random_state=None)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=self.scoring, cv=tscv, verbose=self.verbose, n_jobs=-1)

        try:
            grid_result = grid.fit(processed_x, self.y_train)
            # Storing results in instance variables
            self.best_scores[model_name] = round(grid_result.best_score_, 4)
            self.best_params[model_name] = grid_result.best_params_
            self.all_results[model_name] = {
                'means': [round(mean, 4) for mean in grid_result.cv_results_['mean_test_score']],
                'stds': [round(stdev, 4) for stdev in grid_result.cv_results_['std_test_score']],
                'params': grid_result.cv_results_['params'],
                'ranked_indices': np.argsort(grid_result.cv_results_['rank_test_score'])
            }

            if self.verbose:
                self.print_results()
            # Create performance info string
            best_score_str = f"Best Score: {self.best_scores[model_name]}\n"
            best_params_str = f"Best Parameters: {self.best_params[model_name]}\n"
            performance_info = best_score_str + best_params_str

            # Save the trained model and its performance info
            self._save_model(model_name, grid_result.best_estimator_, performance_info)
        except Exception as e:
            print(f"Error during grid search for model {model_name}: {e}")


if __name__ == "__main__":

    data = load_iris()
    print(data.data[:5])  # Print the first 5 rows of data
    print(data.target[:5])  # Print the first 5 target values
    X = data.data
    y = data.target

    print(len(X), len(y))
    optimizer = GridSearchOptimizer(x_train=X, y_train=y, task_type="classification", scoring='accuracy')

    # Load JSON data from files
    with open("opt_config/classification_config.json", "r") as file:
        classification_data = json.load(file)
    classification_config = AllModelsConfig(**classification_data)

    for model_config in classification_config.models:
        model_name = model_config.model
        model_class = ModelFactory.get_model(model_name, task_type="classification")
        param_grid = model_config.params
        optimizer.run_grid_search(model_name, model_class(), param_grid)