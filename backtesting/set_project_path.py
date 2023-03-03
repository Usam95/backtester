import os
import sys

# Get the absolute path of the project directory
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))

# Add the subfolders of the project to sys.path list
sys.path.append(os.path.join(PROJECT_DIR, '../utilities'))
sys.path.append(os.path.join(PROJECT_DIR, '../hist_data'))
sys.path.append(os.path.join(PROJECT_DIR, '../deployment'))
sys.path.append(os.path.join(PROJECT_DIR, ''))

sys.path.append(os.path.join(PROJECT_DIR, '', 'vectorized_backtesting', 'config'))
sys.path.append(os.path.join(PROJECT_DIR, '', 'vectorized_backtesting', 'backtesting/ml_backtesting'))
sys.path.append(os.path.join(PROJECT_DIR, '', 'vectorized_backtesting', 'strategies'))
