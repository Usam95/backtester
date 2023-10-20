import os


import os
import sys

# Get the absolute path of the project directory
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))

def pathes():
    # Add the subfolders of the project to sys.path list
    sys.path.append(os.path.abspath(os.path.join(PROJECT_DIR, "../..", "utilities")))
    sys.path.append(os.path.abspath(os.path.join(PROJECT_DIR, "../..", "hist_data")))
    sys.path.append(os.path.abspath(os.path.join(PROJECT_DIR, '../..', 'deployment')))
    sys.path.append(os.path.abspath(os.path.join(PROJECT_DIR, "vectorized_backtesting/indicators")))
    sys.path.append(os.path.abspath(os.path.join(PROJECT_DIR, '', 'vectorized_backtesting')))
    sys.path.append(os.path.abspath(os.path.join(PROJECT_DIR, '', 'iterative_backtesting')))
    sys.path.append(os.path.abspath(os.path.join(PROJECT_DIR, '', 'vectorized_backtesting', 'config')))
    sys.path.append(os.path.abspath(os.path.join(PROJECT_DIR, '', 'vectorized_backtesting',
                                                 '../backtesting/ml_backtesting')))


def print_directory_tree(startpath):
    for root, dirs, files in os.walk(startpath):

        # Exclude specific directories
        dirs[:] = [d for d in dirs if d not in [".git", ".idea", "__pycache__", ".ipynb_checkpoints", "results", "tmp", "pdfs", "trained_models"]]

        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)

        # Print directory only if it contains files after exclusions
        if dirs or any(file for file in files if not file.endswith(('.pyc', '__init__.py', '.ipynb', '.gzip'))):
            print(f'{indent}|- {os.path.basename(root)}/')

        sub_indent = ' ' * 4 * (level + 1)

        # Exclude specific file types
        for f in files:
            if not f.endswith(('.pyc', '__init__.py', '.ipynb', '.gzip', '.csv', '.pdf', '.png', '.excalidraw', '.sav')):
                print(f'{sub_indent}|- {f}')


def count_lines_in_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return len(f.readlines())


def count_python_lines(startpath="."):
    total_lines = 0
    for root, dirs, files in os.walk(startpath):
        # Exclude specific directories
        dirs[:] = [d for d in dirs if d not in [".git", ".idea", "__pycache__", ".ipynb_checkpoints"]]

        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                total_lines += count_lines_in_file(file_path)

    return total_lines


project_directory = "."  # current directory
# print_directory_tree(start_directory)
print(f"Total Python lines: {count_python_lines(project_directory)}")
